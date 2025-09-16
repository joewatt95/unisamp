#include <concepts>
#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <ranges>

/**
 * @brief Represents sub-probability measures as a failable, context-dependent
 * computation.
 *
 * This struct models the `ReaderT Maybe` monad transformer. It encapsulates a
 * function that depends on a shared environment (the random number generator
 * `Rng`) which it can mutate, and which may fail to produce a value
 * (`std::optional`). The wrapped function signature is `Rng& ->
 * std::optional<A>`.
 *
 * @tparam A The type of the value to be sampled.
 * @tparam Rng The type of the random number generator (e.g., std::mt19937).
 */
template <typename A, typename Rng = std::mt19937>
struct SubProbMeasure {
  /// The underlying sampler function type.
  using Sampler = std::function<std::optional<A>(Rng&)>;
  const Sampler run;

  /// The type of the value produced by the measure.
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

  // === Core Primitives ===

  /**
   * @brief Lifts a pure value into a SubProbMeasure (monadic return).
   *
   * Creates a measure that always succeeds and returns the given value
   * without consuming any randomness.
   * @param value The value to lift.
   * @return A deterministic SubProbMeasure.
   */
  static SubProbMeasure<A, Rng> pure(const A value) {
    return SubProbMeasure{.run = [v = std::move(value)](Rng&) { return v; }};
  }

  /**
   * @brief Creates a measure that always fails.
   * @return A `SubProbMeasure` that always yields `std::nullopt`.
   */
  static SubProbMeasure<A, Rng> fail() {
    return SubProbMeasure{.run = [](Rng&) { return std::nullopt; }};
  }

  // === Core Samplers ===

  /**
   * @brief A primitive measure for a Bernoulli trial.
   * @param p The probability of success (true).
   * @return A SubProbMeasure that samples a boolean value.
   */
  static SubProbMeasure<bool, Rng> bernoulli(double p) {
    return SubProbMeasure<bool, Rng>{.run = [p](Rng& rng) {
      std::bernoulli_distribution dist(p);
      return dist(rng);
    }};
  }

  /**
   * @brief A primitive measure for a uniform integer distribution.
   * @tparam IntType The integral type to sample (e.g., int, long, size_t).
   * @param min The minimum value of the range (inclusive).
   * @param max The maximum value of the range (inclusive).
   * @return A SubProbMeasure that samples an int.
   */
  template <std::integral IntType = int>
  static SubProbMeasure<IntType, Rng> uniform_int(IntType min, IntType max) {
    return SubProbMeasure<IntType, Rng>{
        .run = [min, max](Rng& rng) -> std::optional<IntType> {
          if (min > max) return std::nullopt;
          std::uniform_int_distribution<IntType> dist(min, max);
          return dist(rng);
        }};
  }

  // --- Compositional Methods ---

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
   * @brief Implements functor map (fmap).
   *
   * Applies a pure function to the result of this measure.
   *
   * @param f A pure function of type `A -> B` to apply to the result.
   * @return A new `SubProbMeasure` with a transformed result value.
   */
  template <typename F>
  auto map(F&& f) const {
    return *this >>= [f = std::forward<F>(f)](const A& a) {
      // Apply the pure function f and lift the result back into a measure.
      return SubProbMeasure<decltype(f(a)), Rng>::pure(f(a));
    };
  }

  /**
   * @brief Creates a measure that fails if a boolean condition is false.
   *
   * This method acts as a monadic guard, a standard pattern in functional
   * programming. It is useful for introducing conditional failure into a
   * chain of computations. If the condition is true, the measure succeeds
   * with a trivial `std::monostate` value, allowing the chain to proceed.
   * If the condition is false, the measure fails, halting the chain.
   *
   * @param condition The boolean condition to check. The measure will fail if
   * this evaluates to `false`.
   * @return A `SubProbMeasure` that succeeds with a trivial value if
   * `condition` is true, and fails otherwise.
   */
  static SubProbMeasure<std::monostate, Rng> guard(bool condition) {
    return SubProbMeasure<std::monostate, Rng>{
        .run = [condition](Rng&) -> std::optional<std::monostate> {
          return condition ? std::optional(std::monostate{}) : std::nullopt;
        }};
  }

  // === Composed Samplers ===

  /**
   * @brief Creates a uniform distribution measure from a multipass range.
   * @tparam Range The type of the input range.
   * @param range The range of elements to sample from.
   * @return A SubProbMeasure representing the uniform distribution.
   */
  template <std::ranges::forward_range Range>
    requires std::is_same_v<std::ranges::range_value_t<Range>, A>
  static SubProbMeasure uniform_range(Range range) {
    if (std::ranges::empty(range)) return SubProbMeasure::fail();

    std::size_t size = std::ranges::distance(range);

    auto sample_index =
        SubProbMeasure<std::size_t, Rng>::uniform_int(std::size_t{0}, size - 1);

    return sample_index.map([r = std::move(range)](std::size_t index) {
      return *std::ranges::next(std::ranges::begin(r), index);
    });
  }

  /**
   * @brief Returns a new measure by scaling this measure's probability.
   * @param factor The scaling factor in the interval [0, 1].
   * @return A new, scaled SubProbMeasure.
   */
  auto scale(double factor) const {
    // Handle edge cases directly for efficiency
    if (factor <= 0.0) return SubProbMeasure::fail();
    if (factor >= 1.0) return *this;

    // 1. Run a Bernoulli trial.
    // 2. Bind the boolean result to a monadic guard that fails if the trial
    // returned false.
    // 3. If the monadic guard passes, bind its trivial result to a continuation
    // that returns the original measure.
    return (SubProbMeasure::bernoulli(factor) >>= &SubProbMeasure::guard) >>=
           [this_measure = *this](const std::monostate&) {
             return this_measure;
           };
  }
};