#pragma once

#include <concepts>
#include <functional>
#include <iterator>
#include <optional>
#include <random>
#include <ranges>

namespace sub_prob_measures {

template <typename Rng>
constexpr auto get_rng() noexcept;

// Forward declarations for factory functions
template <typename A, typename Rng>
constexpr auto pure(const A value) noexcept;

template <typename A, typename Rng>
constexpr auto fail() noexcept;

template <typename Rng>
constexpr auto guard(const bool condition) noexcept;

template <typename Rng>
auto bernoulli(const double p) noexcept;

using RngDefault = std::mt19937;

/**
 * @brief Represents sub-probability measures as a failable, context-dependent
 * computation.
 *
 * This class is modelled after the `MaybeT Reader` monad transformer stack.
 * It encapsulates a function that depends on a shared environment (the random
 * number generator `Rng`) which:
 * 1. Can be any possibly stateful function. Note that for a seeded random
 * number generator, the computation is deterministic.
 * 2. May fail to produce a value (`std::optional`).
 * The wrapped function signature is `Rng& -> std::optional<A>`.
 *
 * @tparam A The type of the value to be sampled.
 * @tparam Rng The type of the random number generator (e.g., std::mt19937).
 */
template <typename A, typename SamplerFunc, typename Rng = RngDefault>
class SubProbMeasure {
 private:
  const SamplerFunc _sampler;

 public:
  /// The type of the value produced by the measure.
  using value_type = A;

  /**
   * @brief Constructs a SubProbMeasure from a callable sampler.
   * @param sampler The callable object that implements the sampling logic.
   */
  constexpr explicit SubProbMeasure(const SamplerFunc&& sampler) noexcept
      : _sampler(std::move(sampler)) {}

  /**
   * @brief Executes the probabilistic computation with a provided RNG.
   * @param rng A reference to the random number generator.
   * @return An optional containing the sampled value, or nullopt on failure.
   */
  constexpr std::optional<A> operator()(Rng& rng) const noexcept {
    return this->_sampler(rng);
  }

  /**
   * @brief Executes the computation using a default, thread-local RNG.
   *
   * If an RNG is not provided, this overload creates and uses a static,
   * thread-local RNG, seeded with std::random_device.
   * @return An optional containing the sampled value, or nullopt on failure.
   */
  std::optional<A> operator()() const noexcept {
    static thread_local Rng default_rng{std::random_device{}()};
    return (*this)(default_rng);
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
  constexpr auto and_then(F&& f) const noexcept {
    using B = typename decltype(f(std::declval<A>()))::value_type;

    const auto new_sampler = [sampler = this->_sampler, f = std::forward<F>(f)](
                                 Rng& rng) noexcept -> std::optional<B> {
      return sampler(rng).and_then(
          [&](const A a) noexcept { return f(std::move(a))(rng); });
    };
    return SubProbMeasure<B, decltype(new_sampler), Rng>(
        std::move(new_sampler));
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
  constexpr auto transform(F&& f) const noexcept {
    using B = decltype(f(std::declval<A>()));

    const auto new_sampler = [sampler = this->_sampler, f = std::forward<F>(f)](
                                 Rng& rng) noexcept -> std::optional<B> {
      // Run the original measure and then transform its optional result.
      return sampler(rng).transform(f);
    };

    return SubProbMeasure<B, decltype(new_sampler), Rng>(
        std::move(new_sampler));
  }

  /**
   * @brief Returns a new measure by scaling this measure's probability.
   * @param factor The scaling factor in the interval [0, 1].
   * @return A new, scaled SubProbMeasure.
   */
  auto scale(const double factor) const noexcept {
    // 1. Run a Bernoulli trial.
    // 2. Bind the boolean result to a monadic guard that fails if the trial
    // returned false.
    // 3. If the monadic guard passes, bind its trivial result to a continuation
    // that returns the original measure.
    return bernoulli<Rng>(factor)
        .and_then(&guard<Rng>)
        .and_then([this_measure = *this](const std::monostate&) noexcept {
          return this_measure;
        });
  }
};

/**
 * @brief Creates a measure that yields the random number generator itself.
 *
 * This function is analogous to the `ask` function in a Reader monad. It
 * creates a computation that, when run, provides access to the environment
 * — in this case, the random number generator (`Rng`). This is useful for
 * building more complex measures that need to inspect or directly manipulate
 * the RNG within a monadic sequence.
 *
 * @tparam Rng The type of the random number generator.
 * @return A `SubProbMeasure` that produces a reference to the current `Rng`.
 */
template <typename Rng = RngDefault>
constexpr auto get_rng() noexcept {
  const auto sampler =
      [](Rng& rng) noexcept -> std::optional<std::reference_wrapper<Rng>> {
    return std::ref(rng);
  };
  return SubProbMeasure<std::reference_wrapper<Rng>, decltype(sampler), Rng>(
      std::move(sampler));
}

/**
 * @brief Lifts a pure value into a SubProbMeasure (monadic return).
 *
 * Creates a measure that always succeeds and returns the given value
 * without consuming any randomness.
 * @param value The value to lift.
 * @return A deterministic SubProbMeasure.
 */
template <typename A, typename Rng = RngDefault>
constexpr auto pure(const A value) noexcept {
  const auto sampler =
      [v = std::move(value)](Rng& /*rng*/) noexcept -> std::optional<A> {
    return v;
  };
  return SubProbMeasure<A, decltype(sampler), Rng>(std::move(sampler));
}

/**
 * @brief Creates a measure that always fails.
 * @return A `SubProbMeasure` that always yields `std::nullopt`.
 */
template <typename A, typename Rng = RngDefault>
constexpr auto fail() noexcept {
  const auto sampler = [](Rng& /*rng*/) noexcept -> std::optional<A> {
    return std::nullopt;
  };
  return SubProbMeasure<A, decltype(sampler), Rng>(std::move(sampler));
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
template <typename Rng = RngDefault>
constexpr auto guard(const bool condition) noexcept {
  const auto sampler =
      [condition](Rng& /*rng*/) noexcept -> std::optional<std::monostate> {
    return condition ? std::optional(std::monostate{}) : std::nullopt;
  };
  return SubProbMeasure<std::monostate, decltype(sampler), Rng>(
      std::move(sampler));
}

/**
 * @brief A primitive measure for a Bernoulli trial.
 * @param p The probability of success (true).
 * @return A SubProbMeasure that samples a boolean value.
 */
template <typename Rng = RngDefault>
auto bernoulli(const double p) noexcept {
  // Get the RNG, then transform it into a boolean result by applying the
  // Bernoulli trial logic.
  return get_rng<Rng>().transform(
      [p](std::reference_wrapper<Rng> rng_ref) noexcept -> bool {
        // Clamp the input p to [0, 1] and handle the trivial cases directly for
        // efficiency.
        if (p <= 0.0) return false;
        if (p >= 1.0) return true;
        std::bernoulli_distribution dist(p);
        return dist(rng_ref.get());
      });
}

/**
 * @brief Creates a measure for a uniform integer distribution.
 * @param min The inclusive lower bound of the range.
 * @param max The inclusive upper bound of the range.
 * @return A measure that produces an `IntType`.
 */
template <std::integral IntType = int, typename Rng = RngDefault>
auto uniform_int(const IntType min, const IntType max) noexcept {
  // Use guard to handle the precondition, then chain the sampling logic.
  return guard<Rng>(min <= max)
      .and_then([min, max](const std::monostate&) noexcept {
        // If the guard passes, get the RNG and perform the sampling.
        return get_rng<Rng>().transform(
            [min,
             max](std::reference_wrapper<Rng> rng_ref) noexcept -> IntType {
              std::uniform_int_distribution<IntType> dist(min, max);
              return dist(rng_ref.get());
            });
      });
}

/**
 * @brief Performs a monadic fold over a range defined by iterators.
 *
 * `fold_m` is a powerful higher-order function that generalizes
 * `std::accumulate`. It threads a state through a sequence of probabilistic
 * computations. For each element in the range, it applies a function `f` that
 * takes the current state and the element, and returns a *measure* that will
 * produce the next state.
 *
 * @tparam It The type of the input iterator.
 * @tparam S The type of the sentinel for the iterator.
 * @tparam State The type of the state being accumulated.
 * @tparam F The type of the step function. Must be callable with
 * `(State, std::iter_value_t<It>)` and return a `SubProbMeasure<State, Rng>`.
 * @param begin The beginning of the range to iterate over.
 * @param end The end of the range.
 * @param init The initial state.
 * @param f The monadic step function.
 * @return A `SubProbMeasure` that, when executed, will produce the final state.
 */
template <std::input_iterator It, std::sentinel_for<It> S, typename State,
          typename F, typename Rng = RngDefault>
auto fold_m(It begin, S end, State init, F&& f) {
  const auto sampler =
      [it = begin, end, init = std::move(init),
       f = std::forward<F>(f)](Rng& rng) {
    auto loop =
        [&](this auto&& self, It current_it,
            std::optional<State> current_state_opt) {
      if (current_it == end || !current_state_opt.has_value()) {
        return current_state_opt;  // Base case: end of range or failure.
      }

      // Recursive step: compute next state and recurse. This is a tail call.
      auto next_state_opt = f(std::move(*current_state_opt), *current_it)(rng);
      return self(std::next(current_it), std::move(next_state_opt));
    };
    return loop(it, std::optional<State>(init));
  };

  return SubProbMeasure<State, decltype(sampler), Rng>(std::move(sampler));
}

/**
 * @brief Creates a uniform distribution measure from a multipass range.
 * @param range The range of elements to sample from.
 * @return A measure that samples one element uniformly from the range. This
 * function supports both single-pass input ranges and multi-pass forward
 * ranges.
 */
template <std::ranges::input_range Range, typename Rng = RngDefault>
auto uniform_range(Range&& range) noexcept {
  using T = std::ranges::range_value_t<Range>;

  if constexpr (std::ranges::forward_range<Range>) {
    // More efficient path for forward ranges (multi-pass capable).
    const auto is_empty = std::ranges::empty(range);
    return guard<Rng>(!is_empty).and_then(
        [r = std::forward<Range>(range)](const std::monostate&) noexcept {
          size_t size;
          if constexpr (std::ranges::sized_range<Range>) {
            size = std::ranges::size(r);
          } else {
            size = std::ranges::distance(r);
          }
          return uniform_int<size_t, Rng>(0, size - 1)
              .transform([r](size_t index) -> T {
                return *std::ranges::next(std::ranges::begin(r), index);
              });
        });
  } else {
    // Path for input ranges (single-pass only) using monadic Reservoir
    // Sampling.
    auto it = std::ranges::begin(range);
    const auto end = std::ranges::end(range);

    if (it == end) return fail<T, Rng>();

    // Define the state for our fold: the reservoir and the current index.
    using FoldState = std::pair<T, size_t>;  // {reservoir, index}

    // Define the step function for the monadic fold.
    auto reservoir_step = [](FoldState state, T current_item) noexcept {
      auto [reservoir, index] = std::move(state);
      const double prob = 1.0 / static_cast<double>(index);

      // Probabilistically decide whether to replace the reservoir.
      return bernoulli<Rng>(prob).transform(
          [reservoir = std::move(reservoir),
           current_item = std::move(current_item),
           index](bool replace) -> FoldState {
            T next_reservoir =
                replace ? std::move(current_item) : std::move(reservoir);
            return {std::move(next_reservoir), index + 1};
          });
    };

    // The initial state is the first item and an index of 2.
    FoldState initial_state = {*it, 2};
    return fold_m<decltype(it), decltype(end), FoldState,
                  decltype(reservoir_step), Rng>(
               std::next(it), end, std::move(initial_state), reservoir_step)
        .transform([](FoldState final_state) { return final_state.first; });
  }
}

}  // namespace sub_prob_measures