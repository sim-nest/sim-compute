use super::*;

#[test]
fn portable_reductions_match_cpu_and_handle_zero_extent() {
    let mut cx = test_cx();
    let source = tensor(&mut cx, vec![2, 3], &["3", "-1", "5", "7", "0.5", "2"]);
    let dtype = Symbol::qualified("numbers", "f32");

    for symbol in [
        sum_op_symbol(),
        min_op_symbol(),
        max_op_symbol(),
        norm_op_symbol(),
    ] {
        let wgpu = execute_portable(
            &mut cx,
            symbol.clone(),
            vec![source.clone()],
            Vec::new(),
            dtype.clone(),
        );
        let cpu = execute_cpu(
            &mut cx,
            symbol,
            vec![source.clone()],
            Vec::new(),
            dtype.clone(),
        );
        assert_same_f32_cells(&wgpu, &cpu);
    }

    let empty = tensor(&mut cx, vec![0, 3], &[]);
    let zero_sum = execute_portable(
        &mut cx,
        sum_op_symbol(),
        vec![empty.clone()],
        Vec::new(),
        dtype.clone(),
    );
    let zero_norm = execute_portable(&mut cx, norm_op_symbol(), vec![empty], Vec::new(), dtype);
    assert_eq!(f32_cells(&zero_sum), vec![0.0]);
    assert_eq!(f32_cells(&zero_norm), vec![0.0]);

    let non_finite = tensor(&mut cx, vec![4], &["NaN", "-1", "inf", "2"]);
    let sum = execute_portable(
        &mut cx,
        sum_op_symbol(),
        vec![non_finite.clone()],
        Vec::new(),
        Symbol::qualified("numbers", "f32"),
    );
    let norm = execute_portable(
        &mut cx,
        norm_op_symbol(),
        vec![non_finite.clone()],
        Vec::new(),
        Symbol::qualified("numbers", "f32"),
    );
    let min = execute_portable(
        &mut cx,
        min_op_symbol(),
        vec![non_finite.clone()],
        Vec::new(),
        Symbol::qualified("numbers", "f32"),
    );
    let max = execute_portable(
        &mut cx,
        max_op_symbol(),
        vec![non_finite],
        Vec::new(),
        Symbol::qualified("numbers", "f32"),
    );
    assert!(f32_cells(&sum)[0].is_nan());
    assert!(f32_cells(&norm)[0].is_nan());
    assert_eq!(f32_cells(&min), vec![-1.0]);
    assert_eq!(f32_cells(&max), vec![f32::INFINITY]);

    let empty_min_op = TensorOp::without_attributes(&mut cx, min_op_symbol()).unwrap();
    let empty_min_input = tensor(&mut cx, vec![0], &[]);
    let empty_min_request = TensorRequest::new(
        empty_min_op,
        vec![empty_min_input],
        TensorMeta::new(Vec::new(), Symbol::qualified("numbers", "f32")),
    );
    let empty_min = execute_portable_kernel(&mut cx, &empty_min_request, WgpuKernelDType::F32);
    match empty_min {
        Ok(_) => panic!("empty min reduction must be rejected"),
        Err(error) => assert!(error.to_string().contains("at least one")),
    }
}

#[test]
fn portable_transpose_dot_and_matmul_match_cpu() {
    let mut cx = test_cx();
    let dtype = Symbol::qualified("numbers", "f32");
    let matrix = tensor(&mut cx, vec![2, 3], &["1", "2", "3", "4", "5", "6"]);
    let transposed = execute_portable(
        &mut cx,
        transpose_exec_op_symbol(),
        vec![matrix.clone()],
        vec![3, 2],
        dtype.clone(),
    );
    assert_same_f32_cells(
        &transposed,
        &execute_cpu(
            &mut cx,
            transpose_exec_op_symbol(),
            vec![matrix.clone()],
            vec![3, 2],
            dtype.clone(),
        ),
    );

    let left_vec = tensor(&mut cx, vec![3], &["1", "2", "3"]);
    let right_vec = tensor(&mut cx, vec![3], &["4", "5", "6"]);
    let dot = execute_portable(
        &mut cx,
        dot_op_symbol(),
        vec![left_vec.clone(), right_vec.clone()],
        Vec::new(),
        dtype.clone(),
    );
    assert_same_f32_cells(
        &dot,
        &execute_cpu(
            &mut cx,
            dot_op_symbol(),
            vec![left_vec.clone(), right_vec.clone()],
            Vec::new(),
            dtype.clone(),
        ),
    );

    let right_matrix = tensor(&mut cx, vec![3, 2], &["7", "8", "9", "10", "11", "12"]);
    let matmul = execute_portable(
        &mut cx,
        matmul_exec_op_symbol(),
        vec![matrix.clone(), right_matrix.clone()],
        vec![2, 2],
        dtype.clone(),
    );
    assert_same_f32_cells(
        &matmul,
        &execute_cpu(
            &mut cx,
            matmul_exec_op_symbol(),
            vec![matrix, right_matrix],
            vec![2, 2],
            dtype.clone(),
        ),
    );

    let matrix_vector_left = tensor(&mut cx, vec![2, 3], &["1", "2", "3", "4", "5", "6"]);
    let matrix_vector = execute_portable(
        &mut cx,
        matmul_exec_op_symbol(),
        vec![matrix_vector_left.clone(), left_vec.clone()],
        vec![2],
        dtype.clone(),
    );
    assert_same_f32_cells(
        &matrix_vector,
        &execute_cpu(
            &mut cx,
            matmul_exec_op_symbol(),
            vec![matrix_vector_left, left_vec],
            vec![2],
            dtype,
        ),
    );
}

#[test]
fn portable_linalg_handles_aliases_segments_oom_and_half_accumulate() {
    let mut cx = test_cx();
    let dtype = Symbol::qualified("numbers", "f32");
    let alias = tensor(&mut cx, vec![3], &["1", "2", "3"]);
    let alias_dot = execute_portable(
        &mut cx,
        dot_op_symbol(),
        vec![alias.clone(), alias],
        Vec::new(),
        dtype.clone(),
    );
    assert_eq!(f32_cells(&alias_dot), vec![14.0]);

    let f16_dtype = Symbol::qualified("numbers", "f16");
    let half_left = tensor(&mut cx, vec![2], &["1.5", "2.5"]);
    let half_right = tensor(&mut cx, vec![2], &["2", "4"]);
    let half = execute_portable(
        &mut cx,
        dot_op_symbol(),
        vec![half_left, half_right],
        Vec::new(),
        f16_dtype,
    );
    assert_eq!(
        parse_f16_literal_cell(&half.cells().unwrap()[0])
            .unwrap()
            .to_f32(),
        half::f16::from_f32(13.0).to_f32()
    );

    let tiny = WgpuTensorExecutor::new(
        WgpuDiscovery::from_probes(
            vec![adapter_with_limits("tiny", "Vulkan", true, limits(4))],
            Vec::new(),
        )
        .adapters[0]
            .clone(),
    );
    let op = TensorOp::without_attributes(&mut cx, add_op_symbol()).unwrap();
    let tiny_left = tensor(&mut cx, vec![2], &["1", "2"]);
    let tiny_right = tensor(&mut cx, vec![2], &["3", "4"]);
    let result = tiny.execute(
        &mut cx,
        TensorRequest::new(
            op,
            vec![tiny_left, tiny_right],
            TensorMeta::new(vec![2], Symbol::qualified("numbers", "f32")),
        ),
    );
    match result {
        Ok(_) => panic!("tiny wgpu profile must reject the submission"),
        Err(error) => assert!(
            error.to_string().contains("device context is unavailable")
                || error.to_string().contains("byte limit")
        ),
    }
}

#[test]
fn physical_reduction_and_linalg_dispatch_match_cpu_when_opted_in() {
    if std::env::var_os("SIM_COMPUTE_WGPU_PHYSICAL").is_none() {
        return;
    }
    let mut cx = test_cx();
    let mut runtimes = discover_wgpu_adapter_runtimes(&Default::default()).unwrap();
    let runtime = runtimes
        .pop()
        .expect("SIM_COMPUTE_WGPU_PHYSICAL requires a probe-backed adapter");
    assert!(verify_physical(&runtime.probe).is_ok());
    let executor = WgpuTensorExecutor::from_parts(
        runtime.probe,
        Some(WgpuExecutionContext {
            device: Arc::new(runtime.device),
            queue: Arc::new(runtime.queue),
        }),
    );
    let dtype = Symbol::qualified("numbers", "f32");
    let source = tensor(&mut cx, vec![2, 3], &["3", "-1", "5", "7", "0.5", "2"]);

    for symbol in [
        sum_op_symbol(),
        min_op_symbol(),
        max_op_symbol(),
        norm_op_symbol(),
    ] {
        let first = execute_wgpu(
            &mut cx,
            &executor,
            symbol.clone(),
            vec![source.clone()],
            Vec::new(),
            dtype.clone(),
        );
        let second = execute_wgpu(
            &mut cx,
            &executor,
            symbol.clone(),
            vec![source.clone()],
            Vec::new(),
            dtype.clone(),
        );
        assert_same_f32_cells(&first, &second);
        assert_same_f32_cells(
            &first,
            &execute_cpu(
                &mut cx,
                symbol,
                vec![source.clone()],
                Vec::new(),
                dtype.clone(),
            ),
        );
        assert!(matches!(
            first.location(),
            TensorLocation::Resident { site, .. } if site == compute_wgpu_site_symbol(0)
        ));
    }

    let matrix = tensor(&mut cx, vec![2, 3], &["1", "2", "3", "4", "5", "6"]);
    let right_matrix = tensor(&mut cx, vec![3, 2], &["7", "8", "9", "10", "11", "12"]);
    let left_vec = tensor(&mut cx, vec![3], &["1", "2", "3"]);
    let right_vec = tensor(&mut cx, vec![3], &["4", "5", "6"]);
    for (symbol, inputs, shape) in [
        (transpose_exec_op_symbol(), vec![matrix.clone()], vec![3, 2]),
        (
            dot_op_symbol(),
            vec![left_vec.clone(), right_vec.clone()],
            Vec::new(),
        ),
        (
            matmul_exec_op_symbol(),
            vec![matrix.clone(), right_matrix.clone()],
            vec![2, 2],
        ),
        (
            matmul_exec_op_symbol(),
            vec![matrix.clone(), left_vec.clone()],
            vec![2],
        ),
        (
            matmul_exec_op_symbol(),
            vec![left_vec.clone(), right_matrix.clone()],
            vec![2],
        ),
    ] {
        let gpu = execute_wgpu(
            &mut cx,
            &executor,
            symbol.clone(),
            inputs.clone(),
            shape.clone(),
            dtype.clone(),
        );
        let cpu = execute_cpu(&mut cx, symbol, inputs, shape, dtype.clone());
        assert_same_f32_cells(&gpu, &cpu);
        assert!(matches!(
            gpu.location(),
            TensorLocation::Resident { site, .. } if site == compute_wgpu_site_symbol(0)
        ));
    }

    let segment_left = tensor(&mut cx, vec![2, 2], &["1", "2", "3", "4"]);
    let segment_right = tensor(&mut cx, vec![2, 3], &["5", "6", "7", "8", "9", "10"]);
    let segmented = execute_wgpu(
        &mut cx,
        &executor,
        matmul_exec_op_symbol(),
        vec![segment_left, segment_right],
        vec![2, 3],
        dtype,
    );
    assert!(
        resident_storage(&segmented)
            .pipeline()
            .as_qualified_str()
            .contains("matmul")
    );
    assert_eq!(executor.flush().unwrap().accepted, 14);
}
