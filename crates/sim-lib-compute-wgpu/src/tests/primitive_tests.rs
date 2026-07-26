use super::*;

#[test]
fn portable_reductions_match_cpu_and_handle_zero_extent() {
    let mut cx = test_cx();
    let executor = WgpuTensorExecutor::new(
        WgpuDiscovery::from_probes(vec![adapter("alpha", "Vulkan", true)], Vec::new()).adapters[0]
            .clone(),
    );
    let source = tensor(&mut cx, vec![2, 3], &["3", "-1", "5", "7", "0.5", "2"]);
    let dtype = Symbol::qualified("numbers", "f32");

    for symbol in [
        sum_op_symbol(),
        min_op_symbol(),
        max_op_symbol(),
        norm_op_symbol(),
    ] {
        let wgpu = execute_wgpu(
            &mut cx,
            &executor,
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
    let zero_sum = execute_wgpu(
        &mut cx,
        &executor,
        sum_op_symbol(),
        vec![empty.clone()],
        Vec::new(),
        dtype.clone(),
    );
    let zero_norm = execute_wgpu(
        &mut cx,
        &executor,
        norm_op_symbol(),
        vec![empty],
        Vec::new(),
        dtype,
    );
    assert_eq!(f32_cells(&zero_sum), vec![0.0]);
    assert_eq!(f32_cells(&zero_norm), vec![0.0]);
}

#[test]
fn portable_transpose_dot_and_matmul_match_cpu() {
    let mut cx = test_cx();
    let executor = WgpuTensorExecutor::new(
        WgpuDiscovery::from_probes(vec![adapter("alpha", "Vulkan", true)], Vec::new()).adapters[0]
            .clone(),
    );
    let dtype = Symbol::qualified("numbers", "f32");
    let matrix = tensor(&mut cx, vec![2, 3], &["1", "2", "3", "4", "5", "6"]);
    let transposed = execute_wgpu(
        &mut cx,
        &executor,
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
    let dot = execute_wgpu(
        &mut cx,
        &executor,
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
    let matmul = execute_wgpu(
        &mut cx,
        &executor,
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
    let matrix_vector = execute_wgpu(
        &mut cx,
        &executor,
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
    let mut small_binding = limits(128);
    small_binding.max_storage_buffer_binding_size = 8;
    let executor = WgpuTensorExecutor::new(
        WgpuDiscovery::from_probes(
            vec![adapter_with_limits("alpha", "Vulkan", true, small_binding)],
            Vec::new(),
        )
        .adapters[0]
            .clone(),
    );
    let dtype = Symbol::qualified("numbers", "f32");
    let alias = tensor(&mut cx, vec![3], &["1", "2", "3"]);
    let alias_dot = execute_wgpu(
        &mut cx,
        &executor,
        dot_op_symbol(),
        vec![alias.clone(), alias],
        Vec::new(),
        dtype.clone(),
    );
    assert_eq!(f32_cells(&alias_dot), vec![14.0]);

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
    assert_eq!(resident_storage(&segmented).segments().len(), 3);
    assert_eq!(
        f32_cells(&segmented),
        vec![21.0, 24.0, 27.0, 47.0, 54.0, 61.0]
    );

    let f16_dtype = Symbol::qualified("numbers", "f16");
    let half_left = tensor(&mut cx, vec![2], &["1.5", "2.5"]);
    let half_right = tensor(&mut cx, vec![2], &["2", "4"]);
    let half = execute_wgpu(
        &mut cx,
        &executor,
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
