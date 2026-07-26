use crate::solver::{ResidentCsrSolver, ResidentKrylovMethod};
use sim_lib_femm_core::{CsrMatrix, FemmError, FemmResult};

const BREAKDOWN_TOL_F32: f32 = 1.0e-20;

#[derive(Clone, Debug)]
pub(crate) struct ResidentCsrMatrix {
    pub(crate) rows: usize,
    rowptr: Vec<usize>,
    colind: Vec<usize>,
    vals: Vec<f32>,
    vals_f64: Vec<f64>,
}

impl ResidentCsrMatrix {
    pub(crate) fn from_csr(matrix: &CsrMatrix) -> FemmResult<Self> {
        matrix.validate()?;
        let vals = matrix
            .vals
            .iter()
            .map(|value| {
                let value = *value as f32;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(FemmError::MalformedMatrix(
                        "resident CSR f32 upload produced a non-finite value".to_owned(),
                    ))
                }
            })
            .collect::<FemmResult<Vec<_>>>()?;
        Ok(Self {
            rows: matrix.rows(),
            rowptr: matrix.rowptr.clone(),
            colind: matrix.colind.clone(),
            vals,
            vals_f64: matrix.vals.clone(),
        })
    }

    pub(crate) fn resident_bytes(&self) -> FemmResult<u64> {
        let rowptr = bytes_for_len(self.rowptr.len(), std::mem::size_of::<usize>())?;
        let colind = bytes_for_len(self.colind.len(), std::mem::size_of::<usize>())?;
        let vals = bytes_for_len(self.vals.len(), std::mem::size_of::<f32>())?;
        Ok(rowptr.saturating_add(colind).saturating_add(vals))
    }

    fn matvec_f32(&self, x: &[f32]) -> FemmResult<Vec<f32>> {
        if x.len() != self.rows {
            return Err(FemmError::MalformedMatrix(format!(
                "resident SpMV vector length {} does not match rows {}",
                x.len(),
                self.rows
            )));
        }
        let mut out = vec![0.0_f32; self.rows];
        for (row, out_value) in out.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for idx in self.rowptr[row]..self.rowptr[row + 1] {
                sum = finite_f32(
                    sum + self.vals[idx] * x[self.colind[idx]],
                    "resident SpMV produced a non-finite value",
                )?;
            }
            *out_value = sum;
        }
        Ok(out)
    }

    pub(crate) fn to_dense_f64(&self) -> FemmResult<Vec<Vec<f64>>> {
        let mut dense = vec![vec![0.0; self.rows]; self.rows];
        for (row, dense_row) in dense.iter_mut().enumerate() {
            for idx in self.rowptr[row]..self.rowptr[row + 1] {
                dense_row[self.colind[idx]] = finite_f64(
                    dense_row[self.colind[idx]] + self.vals_f64[idx],
                    "non-finite f64 correction matrix",
                )?;
            }
        }
        Ok(dense)
    }
}

pub(crate) struct F64Residual {
    pub(crate) values: Vec<f64>,
    pub(crate) norm: f64,
}

pub(crate) fn solve_f32(
    solver: &ResidentCsrSolver,
    matrix: &ResidentCsrMatrix,
    rhs: &[f64],
) -> FemmResult<Vec<f64>> {
    let rhs = rhs
        .iter()
        .map(|value| {
            let value = *value as f32;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(FemmError::MalformedMatrix(
                    "resident RHS f32 upload produced a non-finite value".to_owned(),
                ))
            }
        })
        .collect::<FemmResult<Vec<_>>>()?;
    let x = match solver.config.method {
        ResidentKrylovMethod::Cg => cg_f32(solver, matrix, &rhs)?,
        ResidentKrylovMethod::Bicgstab => bicgstab_f32(solver, matrix, &rhs)?,
    };
    x.into_iter()
        .map(|value| {
            if value.is_finite() {
                Ok(f64::from(value))
            } else {
                Err(FemmError::SolveDidNotConverge(
                    "resident Krylov produced a non-finite solution".to_owned(),
                ))
            }
        })
        .collect()
}

fn cg_f32(
    solver: &ResidentCsrSolver,
    matrix: &ResidentCsrMatrix,
    b: &[f32],
) -> FemmResult<Vec<f32>> {
    if b.iter().all(|value| *value == 0.0) {
        return Ok(vec![0.0; b.len()]);
    }
    let mut x = vec![0.0_f32; b.len()];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old = dot_f32(&r, &r, "resident cg residual")?;
    for iter in 0..solver.config.max_iters {
        let ap = resident_spmv(solver, matrix, &p)?;
        let denom = dot_f32(&p, &ap, "resident cg denominator")?;
        if denom.abs() < BREAKDOWN_TOL_F32 {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident cg breakdown: zero denominator".to_owned(),
            ));
        }
        let alpha = finite_f32(rs_old / denom, "non-finite resident cg alpha")?;
        for i in 0..x.len() {
            x[i] = finite_f32(x[i] + alpha * p[i], "non-finite resident cg solution")?;
            r[i] = finite_f32(r[i] - alpha * ap[i], "non-finite resident cg residual")?;
        }
        resident_vector_update(solver);
        let residual = sync_residual_scalar(solver, &r, iter)?;
        if residual < solver.config.tol_f32 {
            return Ok(x);
        }
        if rs_old.abs() < BREAKDOWN_TOL_F32 {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident cg breakdown: residual denominator vanished".to_owned(),
            ));
        }
        let rs_new = dot_f32(&r, &r, "resident cg residual")?;
        let beta = finite_f32(rs_new / rs_old, "non-finite resident cg beta")?;
        for i in 0..p.len() {
            p[i] = finite_f32(r[i] + beta * p[i], "non-finite resident cg direction")?;
        }
        resident_vector_update(solver);
        rs_old = rs_new;
    }
    solver.refuse(FemmError::SolveDidNotConverge(
        "resident cg did not converge".to_owned(),
    ))
}

fn bicgstab_f32(
    solver: &ResidentCsrSolver,
    matrix: &ResidentCsrMatrix,
    b: &[f32],
) -> FemmResult<Vec<f32>> {
    if b.iter().all(|value| *value == 0.0) {
        return Ok(vec![0.0; b.len()]);
    }
    let n = b.len();
    let mut x = vec![0.0_f32; n];
    let mut r = b.to_vec();
    let r_hat = r.clone();
    let mut rho_prev = 1.0_f32;
    let mut alpha = 1.0_f32;
    let mut omega = 1.0_f32;
    let mut v = vec![0.0_f32; n];
    let mut p = vec![0.0_f32; n];
    for iter in 0..solver.config.max_iters {
        let rho = dot_f32(&r_hat, &r, "resident bicgstab rho")?;
        if rho.abs() < BREAKDOWN_TOL_F32
            || rho_prev.abs() < BREAKDOWN_TOL_F32
            || omega.abs() < BREAKDOWN_TOL_F32
        {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident bicgstab breakdown: update denominator vanished".to_owned(),
            ));
        }
        let beta = finite_f32(
            (rho / rho_prev) * (alpha / omega),
            "non-finite resident bicgstab beta",
        )?;
        for i in 0..n {
            p[i] = finite_f32(
                r[i] + beta * (p[i] - omega * v[i]),
                "non-finite resident bicgstab direction",
            )?;
        }
        resident_vector_update(solver);
        v = resident_spmv(solver, matrix, &p)?;
        let alpha_denom = dot_f32(&r_hat, &v, "resident bicgstab alpha denominator")?;
        if alpha_denom.abs() < BREAKDOWN_TOL_F32 {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident bicgstab breakdown: alpha denominator vanished".to_owned(),
            ));
        }
        alpha = finite_f32(rho / alpha_denom, "non-finite resident bicgstab alpha")?;
        let s = (0..n)
            .map(|i| finite_f32(r[i] - alpha * v[i], "non-finite resident bicgstab stage"))
            .collect::<FemmResult<Vec<_>>>()?;
        resident_vector_update(solver);
        if sync_residual_scalar(solver, &s, iter)? < solver.config.tol_f32 {
            for i in 0..n {
                x[i] = finite_f32(x[i] + alpha * p[i], "non-finite resident bicgstab solution")?;
            }
            return Ok(x);
        }
        let t = resident_spmv(solver, matrix, &s)?;
        let omega_den = dot_f32(&t, &t, "resident bicgstab omega denominator")?;
        if omega_den.abs() < BREAKDOWN_TOL_F32 {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident bicgstab breakdown: omega denominator vanished".to_owned(),
            ));
        }
        omega = finite_f32(
            dot_f32(&t, &s, "resident bicgstab omega numerator")? / omega_den,
            "non-finite resident bicgstab omega",
        )?;
        for i in 0..n {
            x[i] = finite_f32(
                x[i] + alpha * p[i] + omega * s[i],
                "non-finite resident bicgstab solution",
            )?;
            r[i] = finite_f32(s[i] - omega * t[i], "non-finite resident bicgstab residual")?;
        }
        resident_vector_update(solver);
        if sync_residual_scalar(solver, &r, iter)? < solver.config.tol_f32 {
            return Ok(x);
        }
        rho_prev = rho;
    }
    solver.refuse(FemmError::SolveDidNotConverge(
        "resident bicgstab did not converge".to_owned(),
    ))
}

pub(crate) fn resident_spmv(
    solver: &ResidentCsrSolver,
    matrix: &ResidentCsrMatrix,
    x: &[f32],
) -> FemmResult<Vec<f32>> {
    solver
        .state
        .lock()
        .expect("resident CSR state poisoned")
        .snapshot
        .spmv_dispatches += 1;
    matrix.matvec_f32(x)
}

pub(crate) fn resident_vector_update(solver: &ResidentCsrSolver) {
    solver
        .state
        .lock()
        .expect("resident CSR state poisoned")
        .snapshot
        .vector_dispatches += 1;
}

pub(crate) fn sync_residual_scalar(
    solver: &ResidentCsrSolver,
    residual: &[f32],
    iter: usize,
) -> FemmResult<f32> {
    let cadence = solver.config.scalar_sync_cadence.max(1);
    let norm = norm_f32(residual, "resident residual scalar")?;
    if iter.is_multiple_of(cadence) || norm < solver.config.tol_f32 {
        solver
            .state
            .lock()
            .expect("resident CSR state poisoned")
            .snapshot
            .scalar_synchronizations += 1;
    }
    Ok(norm)
}

pub(crate) fn f64_residual(
    matrix: &ResidentCsrMatrix,
    x: &[f64],
    rhs: &[f64],
) -> FemmResult<F64Residual> {
    validate_rhs(matrix.rows, rhs)?;
    if x.len() != matrix.rows {
        return Err(FemmError::MalformedMatrix(format!(
            "solution length {} does not match matrix rows {}",
            x.len(),
            matrix.rows
        )));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(FemmError::SolveDidNotConverge(
            "solution contains a non-finite value".to_owned(),
        ));
    }
    let mut residual = Vec::with_capacity(matrix.rows);
    let mut norm_sq = 0.0_f64;
    for (row, rhs_value) in rhs.iter().enumerate().take(matrix.rows) {
        let mut ax = 0.0_f64;
        for idx in matrix.rowptr[row]..matrix.rowptr[row + 1] {
            ax = finite_f64(
                ax + matrix.vals_f64[idx] * x[matrix.colind[idx]],
                "non-finite f64 residual matvec",
            )?;
        }
        let value = finite_f64(*rhs_value - ax, "non-finite f64 residual")?;
        norm_sq = finite_f64(norm_sq + value * value, "non-finite f64 residual norm")?;
        residual.push(value);
    }
    Ok(F64Residual {
        values: residual,
        norm: finite_f64(norm_sq.sqrt(), "non-finite f64 residual norm")?,
    })
}

pub(crate) fn validate_rhs(rows: usize, rhs: &[f64]) -> FemmResult<()> {
    if rhs.len() != rows {
        return Err(FemmError::MalformedMatrix(format!(
            "rhs length {} does not match matrix rows {rows}",
            rhs.len()
        )));
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(FemmError::MalformedMatrix(
            "rhs values must be finite".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn add_assign_f64(left: &mut [f64], right: &[f64]) -> FemmResult<()> {
    if left.len() != right.len() {
        return Err(FemmError::MalformedMatrix(format!(
            "correction length {} does not match solution length {}",
            right.len(),
            left.len()
        )));
    }
    for (left, right) in left.iter_mut().zip(right) {
        *left = finite_f64(*left + right, "non-finite iterative refinement correction")?;
    }
    Ok(())
}

fn dot_f32(left: &[f32], right: &[f32], context: &str) -> FemmResult<f32> {
    if left.len() != right.len() {
        return Err(FemmError::MalformedMatrix(format!(
            "{context} vector length mismatch: {} != {}",
            left.len(),
            right.len()
        )));
    }
    let mut sum = 0.0_f32;
    for (left, right) in left.iter().zip(right) {
        sum = finite_f32(sum + left * right, context)?;
    }
    Ok(sum)
}

fn norm_f32(values: &[f32], context: &str) -> FemmResult<f32> {
    finite_f32(dot_f32(values, values, context)?.sqrt(), context)
}

fn finite_f32(value: f32, context: &str) -> FemmResult<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FemmError::SolveDidNotConverge(context.to_owned()))
    }
}

fn finite_f64(value: f64, context: &str) -> FemmResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FemmError::SolveDidNotConverge(context.to_owned()))
    }
}

pub(crate) fn bytes_for_len(len: usize, elem: usize) -> FemmResult<u64> {
    let len = u64::try_from(len)
        .map_err(|_| FemmError::BudgetExceeded("resident CSR length exceeds u64".to_owned()))?;
    let elem = u64::try_from(elem).map_err(|_| {
        FemmError::BudgetExceeded("resident CSR element size exceeds u64".to_owned())
    })?;
    len.checked_mul(elem)
        .ok_or_else(|| FemmError::BudgetExceeded("resident CSR byte count overflowed".to_owned()))
}
