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
