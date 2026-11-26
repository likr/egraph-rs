use super::{LeidenObjective, parse_leiden_objective};

#[test]
fn test_leiden_objective_parser() {
    assert_eq!(
        parse_leiden_objective("modularity").unwrap(),
        LeidenObjective::Modularity
    );
    assert_eq!(
        parse_leiden_objective("Mod").unwrap(),
        LeidenObjective::Modularity
    );
    assert_eq!(parse_leiden_objective("CPM").unwrap(), LeidenObjective::Cpm);
    assert!(parse_leiden_objective("louvain").is_err());
}
