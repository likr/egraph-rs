extern crate clp;

#[test]
fn test_resize() {
    let rows = 5;
    let cols = 3;
    let mut model = clp::Model::new();
    model.resize(rows, cols);
    assert_eq!(rows, model.number_rows());
    assert_eq!(cols, model.number_columns());
}

#[test]
fn test_primal() {
    let mut model = clp::Model::new();
    model.set_log_level(0);
    model.set_optimization_direction(-1.0);
    model.resize(2, 0);
    model.add_columns(2,
                      vec![0.0, 0.0],
                      vec![std::f64::INFINITY, std::f64::INFINITY],
                      vec![5.0, 4.0],
                      vec![0, 2, 4],
                      vec![0, 1, 0, 1],
                      vec![5.0, 1.0, 2.0, 2.0]);
    model.change_row_upper(vec![30.0, 14.0]);
    model.primal();
    assert_eq!(model.objective_value(), 40.0);
}
