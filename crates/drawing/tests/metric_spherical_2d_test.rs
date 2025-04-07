use petgraph_drawing::{Delta, DeltaSpherical2d, MetricSpherical2d};

// Helper function to check if a value is NaN
fn is_nan<T: PartialEq + Copy>(value: T) -> bool {
    value != value
}

#[test]
fn test_metric_spherical_2d_basic_operations() {
    // Create two points on the sphere
    let p1 = MetricSpherical2d(0.0f32, 0.0f32); // Point at (1, 0, 0) in Cartesian
    let p2 = MetricSpherical2d(std::f32::consts::PI / 2.0f32, 0.0f32); // Point at (0, 1, 0) in Cartesian

    // Calculate the delta between them
    let delta = &p1 - &p2;

    // The delta should not contain NaN values
    assert!(!is_nan(delta.0), "Delta longitude contains NaN");
    assert!(!is_nan(delta.1), "Delta latitude contains NaN");

    // Create a copy of p1 to test subtraction
    let mut p1_copy = p1;
    p1_copy -= delta;

    // The result should not contain NaN values
    assert!(!is_nan(p1_copy.0), "Result longitude contains NaN");
    assert!(!is_nan(p1_copy.1), "Result latitude contains NaN");
}

#[test]
fn test_metric_spherical_2d_edge_cases() {
    // Test with points at the poles
    let north_pole = MetricSpherical2d(0.0f32, std::f32::consts::PI / 2.0f32);
    let south_pole = MetricSpherical2d(0.0f32, -std::f32::consts::PI / 2.0f32);
    
    // Calculate delta between poles
    let delta = &north_pole - &south_pole;
    
    // The delta should not contain NaN values
    assert!(!is_nan(delta.0), "Delta longitude contains NaN");
    assert!(!is_nan(delta.1), "Delta latitude contains NaN");
    
    // Test with antipodal points on the equator
    let p1 = MetricSpherical2d(0.0f32, 0.0f32);
    let p2 = MetricSpherical2d(std::f32::consts::PI, 0.0f32);
    
    // Calculate delta between antipodal points
    let delta = &p1 - &p2;
    
    // The delta should not contain NaN values
    assert!(!is_nan(delta.0), "Delta longitude contains NaN");
    assert!(!is_nan(delta.1), "Delta latitude contains NaN");
}

#[test]
fn test_metric_spherical_2d_small_movements() {
    // Test with very small movements that might cause numerical issues
    let p1 = MetricSpherical2d(0.0f32, 0.0f32);
    
    // Create a very small delta
    let small_delta = DeltaSpherical2d(1e-10f32, 1e-10f32);
    
    // Apply the small delta
    let mut p1_moved = p1;
    p1_moved -= small_delta;
    
    // The result should not contain NaN values
    assert!(!is_nan(p1_moved.0), "Result longitude contains NaN");
    assert!(!is_nan(p1_moved.1), "Result latitude contains NaN");
    
    // Test with a point very close to another
    let p2 = MetricSpherical2d(1e-10f32, 1e-10f32);
    
    // Calculate delta between very close points
    let delta = &p1 - &p2;
    
    // The delta should not contain NaN values
    assert!(!is_nan(delta.0), "Delta longitude contains NaN");
    assert!(!is_nan(delta.1), "Delta latitude contains NaN");
}

#[test]
fn test_metric_spherical_2d_grid() {
    // Test a grid of points across the sphere to find potential problem areas
    let lon_steps = 8;
    let lat_steps = 4;
    
    for i in 0..lon_steps {
        let lon = i as f32 * 2.0f32 * std::f32::consts::PI / (lon_steps as f32);
        
        for j in 0..lat_steps {
            let lat = j as f32 * std::f32::consts::PI / (lat_steps as f32) - std::f32::consts::PI / 2.0f32;
            
            let p1 = MetricSpherical2d(lon, lat);
            
            // Test against several other points
            for k in 0..lon_steps {
                let other_lon = k as f32 * 2.0f32 * std::f32::consts::PI / (lon_steps as f32);
                
                for l in 0..lat_steps {
                    let other_lat = l as f32 * std::f32::consts::PI / (lat_steps as f32) - std::f32::consts::PI / 2.0f32;
                    
                    let p2 = MetricSpherical2d(other_lon, other_lat);
                    
                    // Calculate delta between points
                    let delta = &p1 - &p2;
                    
                    // The delta should not contain NaN values
                    assert!(!is_nan(delta.0), 
                           "Delta longitude contains NaN for points ({}, {}) and ({}, {})", 
                           lon, lat, other_lon, other_lat);
                    assert!(!is_nan(delta.1), 
                           "Delta latitude contains NaN for points ({}, {}) and ({}, {})", 
                           lon, lat, other_lon, other_lat);
                    
                    // Apply the delta to a copy of p1
                    let mut p1_copy = p1;
                    p1_copy -= delta;
                    
                    // The result should not contain NaN values
                    assert!(!is_nan(p1_copy.0), 
                           "Result longitude contains NaN after applying delta from ({}, {}) to ({}, {})", 
                           lon, lat, other_lon, other_lat);
                    assert!(!is_nan(p1_copy.1), 
                           "Result latitude contains NaN after applying delta from ({}, {}) to ({}, {})", 
                           lon, lat, other_lon, other_lat);
                }
            }
        }
    }
}

#[test]
fn test_metric_spherical_2d_sgd_simulation() {
    // This test simulates the SGD algorithm's operations on spherical coordinates
    // to reproduce the NaN issue seen in the SparseSgd test
    
    // Create two points on the sphere
    let p1 = MetricSpherical2d(0.0f32, 0.0f32);
    let p2 = MetricSpherical2d(std::f32::consts::PI / 4.0f32, std::f32::consts::PI / 4.0f32);
    
    // Calculate the delta between them
    let delta = &p1 - &p2;
    
    // Calculate the norm of the delta
    let norm = delta.norm();
    
    // Simulate the SGD algorithm's operations
    if norm > 0.0f32 {
        // Target distance
        let target_distance = 0.5f32;
        
        // Calculate the scaling factor
        let r = 0.5f32 * (norm - target_distance) / norm;
        
        // Learning rate
        let eta = 0.1f32;
        let mu = (eta * 1.0f32).min(1.0f32);
        
        // Apply the scaled delta to p1
        let mut p1_moved = p1;
        p1_moved -= delta * (-r * mu);
        
        // The result should not contain NaN values
        assert!(!is_nan(p1_moved.0), "Result longitude contains NaN after SGD-like operation");
        assert!(!is_nan(p1_moved.1), "Result latitude contains NaN after SGD-like operation");
        
        // Apply the scaled delta to p2
        let mut p2_moved = p2;
        p2_moved -= delta * (r * mu);
        
        // The result should not contain NaN values
        assert!(!is_nan(p2_moved.0), "Result longitude contains NaN after SGD-like operation");
        assert!(!is_nan(p2_moved.1), "Result latitude contains NaN after SGD-like operation");
    }
}
