#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <ranges>

/**
 * @brief Represents a sub-probability measure.
 *
 * A SubProbMeasure encapsulates a probabilistic function that takes a random
 * number generator (Rng) and returns an optional value of type A. It models
 * computations that depend on an RNG and can fail (return nullopt).
 *
 * @tparam A The type of the value to be sampled.
 * @tparam Rng The type of the random number generator.
 */
template <typename A, typename Rng = std::mt19937>
struct SubProbMeasure {
  // The underlying sampler function type.
  using Sampler = std::function<std::optional<A>(Rng&)>;
  const Sampler run;

  // The type of the value produced by the measure.
  using value_type = A;

  /**
   * @brief Executes the probabilistic computation with a provided RNG.
   * @param rng A reference to the random number generator.
   * @return An optional containing the sampled value, or nullopt on failure.
   */
  std::optional<A> operator()(Rng& rng) const { return run(rng); }

  /**
   * @brief Executes the computation using a default, thread-local RNG.
   *
   * If an RNG is not provided, this overload creates and uses a static,
   * thread-local RNG, seeded with std::random_device.
   * @return An optional containing the sampled value, or nullopt on failure.
   */
  std::optional<A> operator()() const {
    static thread_local Rng default_rng{std::random_device{}()};
    return run(default_rng);
  }

  /**
   * @brief Implements Kleisli composition (monadic bind).
   *
   * Composes this measure with a function `f` that takes a value of type `A`
   * and returns a new SubProbMeasure. The mathematical formulation is:
   * (m >>= f) rng = match (m rng) with
   * | None -> None
   * | Some(a) -> (f a) rng
   *
   * @tparam F A callable type mapping A to SubProbMeasure<B, Rng>.
   * @param f The function to compose with.
   * @return A new SubProbMeasure representing the composed computation.
   */
  template <typename F>
  auto operator>>=(F&& f) const {
    // Deduce the output type B from the function f
    using B = typename decltype(f(std::declval<A>()))::value_type;

    return SubProbMeasure<B, Rng>{
        .run = [this_run = this->run,
                f = std::forward<F>(f)](Rng& rng) -> std::optional<B> {
          // 1. Run the first measure to get an optional<A>
          if (auto opt_a = this_run(rng)) {
            // 2. If it succeeds, apply f to get the next measure
            // 3. Run the next measure with the same rng
            return f(*opt_a)(rng);
          }
          // 4. If the first measure fails, the composition fails
          return std::nullopt;
        }};
  }

  /**
   * @brief Returns a new measure by scaling this measure's probability.
   *
   * The new measure samples from this measure with a probability of `scale`.
   * With probability `1 - scale`, it returns nullopt.
   *
   * @param scale A scaling factor in the interval [0, 1].
   * - If scale <= 0, the resulting measure always returns nullopt.
   * - If scale >= 1, the resulting measure is identical to this one.
   * @return A new, scaled SubProbMeasure.
   */
  auto scale(double scale) const {
    if (scale <= 0.0)
      return SubProbMeasure<A, Rng>{.run = [](Rng&) { return std::nullopt; }};

    if (scale >= 1.0) return *this;

    return SubProbMeasure<A, Rng>{
        .run = [scale, this_run = this->run](Rng& rng) -> std::optional<A> {
          std::bernoulli_distribution dist(scale);
          if (dist(rng)) {
            return this_run(rng);  // Sample from the original measure
          } else {
            return std::nullopt;  // Fail
          }
        }};
  }
};

/**
 * @brief Creates a sub-probability measure for a uniform distribution over a
 * range.
 *
 * @tparam Range A multipass range type (e.g., std::vector, std::list).
 * @tparam Rng The random number generator type.
 * @param range The input range of elements to sample from.
 * @return A SubProbMeasure that, when run, samples one element uniformly
 * from the range. Returns nullopt if the range is empty.
 */
template <std::ranges::forward_range Range, typename Rng = std::mt19937>
auto uniform_prob_measure(Range range) {
  using T = std::ranges::range_value_t<Range>;

  return SubProbMeasure<T, Rng>{
      .run = [r = std::move(range)](Rng& rng) -> std::optional<T> {
        if (std::ranges::empty(r)) {
          return std::nullopt;
        }
        const auto size = std::ranges::distance(r);
        std::uniform_int_distribution<std::size_t> dist(0, size - 1);
        return *std::ranges::next(std::ranges::begin(r), dist(rng));
      }};
}