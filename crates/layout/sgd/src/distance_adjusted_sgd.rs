use crate::Sgd;
use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};
use std::collections::HashMap;

/// Distance-Adjusted Stochastic Gradient Descent (SGD) implementation for graph layout.
///
/// This implementation wraps another SGD algorithm and adjusts the target distances
/// during the layout process based on the current geometric distances. This can help
/// prevent node overlap and create more aesthetically pleasing layouts by allowing
/// some flexibility in the target distances.
///
/// The adjustment is controlled by an alpha parameter that balances between the original
/// graph-theoretic distances and the current geometric distances.
pub struct DistanceAdjustedSgd<A, S>
where
    A: Sgd<S>,
{
    /// Controls the balance between original distances and current geometric distances.
    /// - Values closer to 0 favor the original distances
    /// - Values closer to 1 allow more adjustment based on current layout
    pub alpha: S,

    /// Minimum allowed distance between nodes after adjustment.
    /// This helps prevent nodes from overlapping or getting too close.
    pub minimum_distance: S,

    /// The underlying SGD algorithm being wrapped
    sgd: A,

    /// Map of original distances between node pairs, used as a reference
    /// during the distance adjustment process
    original_distance: HashMap<(usize, usize), S>,
}

impl<A, S> DistanceAdjustedSgd<A, S>
where
    A: Sgd<S>,
{
    /// Creates a new DistanceAdjustedSgd instance wrapping the provided SGD algorithm.
    ///
    /// This constructor initializes the distance adjustment parameters with default values
    /// and stores the original distances from the wrapped SGD algorithm for later reference.
    ///
    /// # Parameters
    /// * `sgd` - The underlying SGD algorithm to wrap
    ///
    /// # Returns
    /// A new DistanceAdjustedSgd instance with:
    /// - alpha = 0.5 (balanced adjustment)
    /// - minimum_distance = 0.0 (no minimum enforced by default)
    pub fn new(sgd: A) -> DistanceAdjustedSgd<A, S>
    where
        S: DrawingValue,
    {
        let mut original_distance = HashMap::new();
        for p in sgd.node_pairs().iter() {
            original_distance.insert((p.0, p.1), p.2);
        }
        Self {
            alpha: S::from_f32(0.5).unwrap(),
            minimum_distance: S::from(0.0).unwrap(),
            sgd,
            original_distance,
        }
    }

    /// Applies the SGD layout algorithm with dynamic distance adjustment.
    ///
    /// This method:
    /// 1. First applies the standard SGD iteration using the wrapped algorithm
    /// 2. Updates the target distances based on the current geometric distances
    /// 3. Updates the weights based on the new target distances
    ///
    /// The distance adjustment uses a weighted average between current geometric
    /// distances and original graph-theoretic distances, controlled by the alpha parameter.
    ///
    /// # Parameters
    /// * `drawing` - The current node position drawing to update
    /// * `eta` - The current learning rate
    pub fn apply_with_distance_adjustment<D, Diff, M>(&mut self, drawing: &mut D, eta: S)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: Metric<D = Diff>,
        S: DrawingValue,
    {
        self.sgd.apply(drawing, eta);
        self.sgd.update_distance(|i, j, _, w| {
            let delta = drawing.delta(i, j);
            let d1 = delta.norm();
            let d2 = self.original_distance[&(i, j)];
            let new_d = (self.alpha * w * d1
                + S::from_usize(2).unwrap() * (S::one() - self.alpha) * d2)
                / (self.alpha * w + S::from_usize(2).unwrap() * (S::one() - self.alpha));
            new_d.max(self.minimum_distance).min(d2)
        });
        self.sgd.update_weight(|_, _, d, _| S::one() / (d * d));
    }
}

/// Implementation of the Sgd trait for DistanceAdjustedSgd
///
/// This delegates the core SGD functionality to the wrapped algorithm instance,
/// allowing the distance-adjusted algorithm to be used within the common SGD framework.
impl<A, S> Sgd<S> for DistanceAdjustedSgd<A, S>
where
    A: Sgd<S>,
{
    /// Returns a reference to the node pairs from the wrapped SGD algorithm.
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S, S)> {
        self.sgd.node_pairs()
    }

    /// Returns a mutable reference to the node pairs from the wrapped SGD algorithm.
    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S, S)> {
        self.sgd.node_pairs_mut()
    }
}
