use sim_lib_femm_core::{CsrMatrix, FemmError, FemmResult};
use sim_lib_numbers_tensor::{Tensor, matmul_exec_op_symbol};

use crate::provider_work::ProviderWork;
use crate::solver::{ResidentCsrSolver, ResidentKrylovMethod};

const BREAKDOWN_TOL_F32: f32 = 1.0e-20;

#[derive(Clone)]
pub(crate) struct ResidentCsrMatrix {
    pub(crate) rows: usize,
    rowptr: Vec<usize>,
    colind: Vec<usize>,
    vals: Vec<f32>,
    vals_f64: Vec<f64>,
    resident_dense: Tensor,
}

impl ResidentCsrMatrix {
    pub(crate) fn from_csr(solver: &ResidentCsrSolver, matrix: &CsrMatrix) -> FemmResult<Self> {
        matrix.validate()?;
        let rows = matrix.rows();
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
        let mut dense = vec![0.0_f32; rows * rows];
        for row in 0..rows {
            for idx in matrix.rowptr[row]..matrix.rowptr[row + 1] {
                dense[row * rows + matrix.colind[idx]] = vals[idx];
            }
        }
        let mut work = ProviderWork::new(solver);
        let resident_dense = work.upload_tensor(vec![rows, rows], &dense)?;
        work.finish()?;
        Ok(Self {
            rows,
            rowptr: matrix.rowptr.clone(),
            colind: matrix.colind.clone(),
            vals,
            vals_f64: matrix.vals.clone(),
            resident_dense,
        })
    }

    pub(crate) fn resident_bytes(&self) -> FemmResult<u64> {
        let rowptr = bytes_for_len(self.rowptr.len(), std::mem::size_of::<usize>())?;
        let colind = bytes_for_len(self.colind.len(), std::mem::size_of::<usize>())?;
        let vals = bytes_for_len(self.vals.len(), std::mem::size_of::<f32>())?;
        Ok(rowptr.saturating_add(colind).saturating_add(vals))
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
    let mut work = ProviderWork::new(solver);
    let rhs = work.upload_tensor(vec![rhs.len()], &rhs)?;
    let x = match solver.config.method {
        ResidentKrylovMethod::Cg => cg_f32(solver, &mut work, matrix, &rhs)?,
        ResidentKrylovMethod::Bicgstab => bicgstab_f32(solver, &mut work, matrix, &rhs)?,
    };
    let x = work.tensor_values(&x)?;
    work.finish()?;
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
    work: &mut ProviderWork<'_>,
    matrix: &ResidentCsrMatrix,
    b: &Tensor,
) -> FemmResult<Tensor> {
    let b_values = work.tensor_values(b)?;
    if b_values.iter().all(|value| *value == 0.0) {
        return work.upload_tensor(vec![b_values.len()], &vec![0.0; b_values.len()]);
    }
    let mut x = work.upload_tensor(vec![b_values.len()], &vec![0.0; b_values.len()])?;
    let mut r = b.clone();
    let mut p = r.clone();
    let mut rs_old = work.dot(&r, &r, "resident cg residual")?;
    for iter in 0..solver.config.max_iters {
        let ap = resident_spmv(solver, work, matrix, &p)?;
        let denom = work.dot(&p, &ap, "resident cg denominator")?;
        if denom.abs() < BREAKDOWN_TOL_F32 {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident cg breakdown: zero denominator".to_owned(),
            ));
        }
        let alpha = finite_f32(rs_old / denom, "non-finite resident cg alpha")?;
        let alpha_p = work.scale(&p, alpha)?;
        x = work.add(&x, &alpha_p, "non-finite resident cg solution")?;
        let alpha_ap = work.scale(&ap, alpha)?;
        r = work.sub(&r, &alpha_ap, "non-finite resident cg residual")?;
        resident_vector_update(solver, 2);
        let residual = sync_residual_scalar(solver, work, &r, iter)?;
        if residual < solver.config.tol_f32 {
            return Ok(x);
        }
        if rs_old.abs() < BREAKDOWN_TOL_F32 {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident cg breakdown: residual denominator vanished".to_owned(),
            ));
        }
        let rs_new = work.dot(&r, &r, "resident cg residual")?;
        let beta = finite_f32(rs_new / rs_old, "non-finite resident cg beta")?;
        let beta_p = work.scale(&p, beta)?;
        p = work.add(&r, &beta_p, "non-finite resident cg direction")?;
        resident_vector_update(solver, 1);
        rs_old = rs_new;
    }
    solver.refuse(FemmError::SolveDidNotConverge(
        "resident cg did not converge".to_owned(),
    ))
}

fn bicgstab_f32(
    solver: &ResidentCsrSolver,
    work: &mut ProviderWork<'_>,
    matrix: &ResidentCsrMatrix,
    b: &Tensor,
) -> FemmResult<Tensor> {
    let b_values = work.tensor_values(b)?;
    if b_values.iter().all(|value| *value == 0.0) {
        return work.upload_tensor(vec![b_values.len()], &vec![0.0; b_values.len()]);
    }
    let n = b_values.len();
    let mut x = work.upload_tensor(vec![n], &vec![0.0; n])?;
    let mut r = b.clone();
    let r_hat = r.clone();
    let mut rho_prev = 1.0_f32;
    let mut alpha = 1.0_f32;
    let mut omega = 1.0_f32;
    let mut v = work.upload_tensor(vec![n], &vec![0.0; n])?;
    let mut p = work.upload_tensor(vec![n], &vec![0.0; n])?;
    for iter in 0..solver.config.max_iters {
        let rho = work.dot(&r_hat, &r, "resident bicgstab rho")?;
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
        let omega_v = work.scale(&v, omega)?;
        let p_minus_omega_v = work.sub(&p, &omega_v, "non-finite resident bicgstab direction")?;
        let beta_direction = work.scale(&p_minus_omega_v, beta)?;
        p = work.add(
            &r,
            &beta_direction,
            "non-finite resident bicgstab direction",
        )?;
        resident_vector_update(solver, 3);
        v = resident_spmv(solver, work, matrix, &p)?;
        let alpha_denom = work.dot(&r_hat, &v, "resident bicgstab alpha denominator")?;
        if alpha_denom.abs() < BREAKDOWN_TOL_F32 {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident bicgstab breakdown: alpha denominator vanished".to_owned(),
            ));
        }
        alpha = finite_f32(rho / alpha_denom, "non-finite resident bicgstab alpha")?;
        let alpha_v = work.scale(&v, alpha)?;
        let s = work.sub(&r, &alpha_v, "non-finite resident bicgstab stage")?;
        resident_vector_update(solver, 2);
        if sync_residual_scalar(solver, work, &s, iter)? < solver.config.tol_f32 {
            let alpha_p = work.scale(&p, alpha)?;
            x = work.add(&x, &alpha_p, "non-finite resident bicgstab solution")?;
            return Ok(x);
        }
        let t = resident_spmv(solver, work, matrix, &s)?;
        let omega_den = work.dot(&t, &t, "resident bicgstab omega denominator")?;
        if omega_den.abs() < BREAKDOWN_TOL_F32 {
            return solver.refuse(FemmError::SolveDidNotConverge(
                "resident bicgstab breakdown: omega denominator vanished".to_owned(),
            ));
        }
        omega = finite_f32(
            work.dot(&t, &s, "resident bicgstab omega numerator")? / omega_den,
            "non-finite resident bicgstab omega",
        )?;
        let alpha_p = work.scale(&p, alpha)?;
        let omega_s = work.scale(&s, omega)?;
        let x_stage = work.add(&x, &alpha_p, "non-finite resident bicgstab solution")?;
        x = work.add(&x_stage, &omega_s, "non-finite resident bicgstab solution")?;
        let omega_t = work.scale(&t, omega)?;
        r = work.sub(&s, &omega_t, "non-finite resident bicgstab residual")?;
        resident_vector_update(solver, 4);
        if sync_residual_scalar(solver, work, &r, iter)? < solver.config.tol_f32 {
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
    work: &mut ProviderWork<'_>,
    matrix: &ResidentCsrMatrix,
    x: &Tensor,
) -> FemmResult<Tensor> {
    solver
        .state
        .lock()
        .expect("resident CSR state poisoned")
        .snapshot
        .spmv_dispatches += 1;
    work.execute(
        matmul_exec_op_symbol(),
        vec![matrix.resident_dense.clone(), x.clone()],
        vec![matrix.rows],
        "resident SpMV",
    )
}

pub(crate) fn resident_vector_update(solver: &ResidentCsrSolver, count: usize) {
    solver
        .state
        .lock()
        .expect("resident CSR state poisoned")
        .snapshot
        .vector_dispatches += count;
}

pub(crate) fn sync_residual_scalar(
    solver: &ResidentCsrSolver,
    work: &mut ProviderWork<'_>,
    residual: &Tensor,
    iter: usize,
) -> FemmResult<f32> {
    let cadence = solver.config.scalar_sync_cadence.max(1);
    let norm = work.norm(residual, "resident residual scalar")?;
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
