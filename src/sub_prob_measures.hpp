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
  constexpr std::optional<A> operator()(Rng& rng) const
      noexcept(noexcept(this->_sampler(rng))) {
    return this->_sampler(rng);
  }

  /**
   * @brief Executes the computation using a default, thread-local RNG.
   *
   * If an RNG is not provided, this overload uses a static, thread-local RNG.
   * Note: The first call on a thread may throw if `std::random_device`
   * fails to acquire entropy to seed the default RNG.
   * @return An optional containing the sampled value, or nullopt on failure.
   */
  std::optional<A> operator()() const {
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
  constexpr auto and_then(F&& f) const
      noexcept(noexcept(f(std::declval<A>()))) {
    using B = typename decltype(f(std::declval<A>()))::value_type;

    const auto new_sampler =
        [sampler = this->_sampler, f = std::forward<F>(f)](Rng& rng) noexcept(
            noexcept(f(std::declval<A>()))) -> std::optional<B> {
      return sampler(rng).and_then(
          [&](const A a) noexcept(noexcept(f(std::declval<A>()))) {
            return f(std::move(a))(rng);
          });
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
  constexpr auto transform(F&& f) const
      noexcept(noexcept(f(std::declval<A>()))) {
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
constexpr auto pure(A value) noexcept {
  auto sampler =
      [v = std::move(value)](Rng& /*rng*/) noexcept -> std::optional<A> {
    return std::move(v);
  };
  return SubProbMeasure<A, decltype(sampler), Rng>(std::move(sampler));
}

// constexpr auto pure(const A value) noexcept {
//   const auto sampler =
//       [v = std::move(value)](Rng& /*rng*/) noexcept -> std::optional<A> {
//     return v;
//   };
//   return SubProbMeasure<A, decltype(sampler), Rng>(std::move(sampler));
// }

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

// Concept to check if Dist is a valid numeric distribution for NumType.
template <template <typename> class Dist, typename NumType, typename Rng>
concept NumericDistribution =
    (std::integral<NumType> || std::floating_point<NumType>) &&
    requires(Rng& rng, NumType val) {
      // Check that Dist<NumType> can be constructed and called
      // with an RNG.
      { Dist<NumType>(val, val)(rng) } -> std::convertible_to<NumType>;
    };

// Helper to determine if a distribution call is nothrow.
template <template <typename> class Dist, typename NumType>
constexpr bool is_dist_call_nothrow_v =
    noexcept(Dist<NumType>(std::declval<NumType>(), std::declval<NumType>())(
        std::declval<RngDefault&>()));

/**
 * @brief Generic factory for creating numeric distribution measures.
 *
 * This internal helper function abstracts the common pattern of guarding,
 * chaining, and sampling used by both integer and real distributions.
 *
 * @tparam Dist The numeric distribution class template (e.g.,
 * `std::uniform_int_distribution`).
 * @tparam NumType The numeric type to be sampled.
 * @tparam Rng The random number generator type.
 */
template <template <typename> class Dist, typename NumType, typename Rng>
  requires NumericDistribution<Dist, NumType, Rng>
auto numeric_dist(const NumType min, const NumType max) noexcept {
  // Use guard to handle the precondition, then chain the sampling logic.
  return guard<Rng>(min <= max)
      .and_then([min, max](const std::monostate&) noexcept {
        // If the guard passes, get the RNG and perform the sampling.
        return get_rng<Rng>().transform(
            [min, max](std::reference_wrapper<Rng> rng_ref) noexcept(
                is_dist_call_nothrow_v<Dist, NumType>) {
              return Dist<NumType>(min, max)(rng_ref.get());
            });
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
  return numeric_dist<std::uniform_int_distribution, IntType, Rng>(min, max);
}

/**
 * @brief Creates a measure for a uniform real distribution.
 * @param min The inclusive lower bound of the range.
 * @param max The exclusive upper bound of the range.
 * @return A measure that produces a `RealType`.
 */
template <std::floating_point RealType = double, typename Rng = RngDefault>
auto uniform_real(const RealType min, const RealType max) noexcept {
  return numeric_dist<std::uniform_real_distribution, RealType, Rng>(min, max);
}

// Helper variable template to determine if the gfold_m/fold_m factory is
// nothrow. This checks if all captures into the sampler lambda are
// nothrow-constructible.
template <typename It, typename S, typename State, typename F>
constexpr bool is_gfold_factory_nothrow_v =
    std::is_nothrow_move_constructible_v<It> &&  // Check move, not copy
    std::is_nothrow_constructible_v<S, S&> &&
    std::is_nothrow_move_constructible_v<State> &&
    std::is_nothrow_constructible_v<std::decay_t<F>, F>;

/**
 * @brief Performs a generalized monadic fold over a range.
 *
 * This is a powerful higher-order function for stateful, probabilistic
 * iteration. Unlike a standard fold that processes every element, `gfold_m`
 * gives the step function `f` full control over how the iterator advances.
 * This allows for complex iteration patterns, such as skipping multiple
 * elements, which is used to implement Algorithm L for Reservoir Sampling.
 *
 * @tparam It The type of the input iterator for the range.
 * @tparam S The type of the sentinel for the iterator.
 * @tparam State The type of the state being accumulated.
 * @tparam F The type of the step function. Must be callable with
 * `(State current_state, It current_iterator)` and return a
 * `SubProbMeasure<std::pair<State, It>>`. The returned measure, upon success,
 * must yield a pair containing the next state and the next iterator position.
 * @tparam Rng The type of the random number generator.
 * @param begin The beginning of the range to iterate over.
 * @param end The end of the range.
 * @param init The initial state.
 * @param f The generalized monadic step function.
 * @return A `SubProbMeasure` that produces the final state.
 */
template <std::input_iterator It, std::sentinel_for<It> S, typename State,
          typename F, typename Rng = RngDefault>
auto gfold_m(It&& begin, S end, State init,
             F&& f) noexcept(is_gfold_factory_nothrow_v<It, S, State, F>) {
  // The sampler and loop are noexcept if the step function `f` is noexcept.
  static constexpr bool is_step_nothrow =
      noexcept(f(std::declval<State>(), std::declval<It>()));

  const auto sampler =
      [it = std::move(begin), end, init = std::move(init),
       f = std::forward<F>(f)](Rng& rng) noexcept(is_step_nothrow) {
        // Tail-recursive loop. Returns the final state upon successful
        // completion, or nullopt if any step fails. The iterator is passed by
        // rvalue reference to support move-only iterators.
        const auto loop = [&](this auto&& self, auto&& current_it,
                              auto&& current_state) noexcept(is_step_nothrow) {
          // Base case: end of range, return the accumulated state.
          if (current_it == end) return current_state;

          // Execute the step function to get the measure for the next state and
          // iterator.
          using CurrentIt = std::decay_t<decltype(current_it)>;
          using NextPair =
              std::pair<std::decay_t<decltype(current_state)>, CurrentIt>;
          auto next_pair_opt =
              f(std::forward<decltype(current_state)>(current_state),
                std::forward<decltype(current_it)>(current_it))(rng);
          return next_pair_opt.and_then(
              [&](NextPair&& next_pair) noexcept(is_step_nothrow) {
                auto [next_state, next_it] = std::move(next_pair);
                return self(std::move(next_it), std::move(next_state));
              });
        };

        // The loop returns the final state if it runs to completion. If it
        // terminates early (returning nullopt), we know the *previous* state
        // (`init` in this first call) was the last valid one.
        return loop(std::move(it), std::move(init));
      };

  return SubProbMeasure<State, decltype(sampler), Rng>(std::move(sampler));
}

/**
 * @brief Performs a monadic fold over a range, processing one element at a
 * time.
 *
 * `fold_m` is a powerful higher-order function that generalizes
 * `std::accumulate`. It threads a state through a sequence of probabilistic
 * computations.
 * This function is implemented as a specialization of `gfold_m`, where the
 * iterator is advanced by one at each step.
 *
 * @tparam It The type of the input iterator for the range.
 * @tparam S The type of the sentinel for the iterator.
 * @tparam State The type of the state being accumulated.
 * @tparam F The type of the step function. Must be callable with
 * `(State current_state, std::iter_value_t<It> current_value)` and return a
 * `SubProbMeasure<State>`.
 * @tparam Rng The type of the random number generator.
 * @param begin The beginning of the range to iterate over.
 * @param end The end of the range.
 * @param init The initial state.
 * @param f The monadic step function.
 * @return A `SubProbMeasure` that produces the final accumulated state.
 */
// template <std::input_iterator It, std::sentinel_for<It> S, typename State,
//           typename F, typename Rng = RngDefault>
// auto fold_m(It&& begin, S end, State init,
//             F&& f) noexcept(is_gfold_factory_nothrow_v<It, S, State, F>) {
//   // Define a step function for gfold_m that advances the iterator by one.
//   static constexpr bool is_step_nothrow =
//       noexcept(f(std::declval<State>(), std::declval<std::iter_value_t<It>>()));

//   const auto gfold_step = [f = std::forward<F>(f)](
//                               State current_state,
//                               It current_it) noexcept(is_step_nothrow) {
//     using NextPair = std::pair<State, It>;

//     // Apply the user's function `f` and then transform the resulting measure.
//     // The transformation wraps the new state in a pair with the next iterator.
//     return f(std::move(current_state), *current_it)
//         .transform([current_it = std::move(current_it)](
//                        State next_state) mutable -> NextPair {
//           return {std::move(next_state),
//                   std::move(std::next(std::move(current_it)))};
//         });
//   };

//   return gfold_m<It, S, State, decltype(gfold_step), Rng>(
//       std::move(begin), end, std::move(init), std::move(gfold_step));
// }

/**
 * @brief Creates a measure to sample one element uniformly from a range.
 *
 * This function provides two specialized implementations based on the range's
 * capabilities:
 * 1.  For multi-pass ranges (e.g., `std::vector`, `std::list`), it can
 *     calculate the range's size and directly sample a random index. This is
 *     highly efficient.
 * 2.  For single-pass input ranges (e.g., an input stream, a filter view),
 *     where the size is unknown, it uses **Algorithm L for Reservoir
 *     Sampling**. This advanced algorithm processes the stream in a single
 *     pass and is highly efficient for large streams because it can
 *     probabilistically calculate how many elements to skip, avoiding the need
 *     to process every single one. The implementation elegantly expresses this
 *     complex iteration pattern using the generalized monadic fold `gfold_m`.
 *
 * @param range The range of elements to sample from.
 * @return A measure that samples one element uniformly from the range.
 */
template <std::ranges::input_range Range, typename Rng = RngDefault>
auto uniform_range(Range&& range) noexcept {
  using T = std::ranges::range_value_t<Range>;

  if constexpr (std::ranges::forward_range<Range>) {
    // More efficient path for forward ranges (multi-pass capable).
    const auto is_empty = std::ranges::empty(range);
    return guard<Rng>(!is_empty).and_then(
        [r = std::forward<Range>(range)](const std::monostate&) noexcept {
          const size_t size = [&r] {
            if constexpr (std::ranges::sized_range<Range>) {
              return std::ranges::size(r);
            } else {
              return std::ranges::distance(r);
            }
          }();
          return uniform_int<size_t, Rng>(0, size - 1)
              .transform([r](size_t index) -> T {
                return *std::ranges::next(std::ranges::begin(r), index);
              });
        });
  } else {
    // Path for input ranges (single-pass only), via Algorithm L for reservoir
    // sampling.
    auto it = std::ranges::begin(range);
    const auto end = std::ranges::end(range);
    if (it == end) {
      return fail<T, Rng>();
    }

    // using ItType = decltype(it);
    struct LoopState {
      T reservoir;
      size_t items_seen;
    };

    // This is the step function for gfold_m that implements one iteration of
    // Algorithm L. It takes the current state (reservoir and count) and
    // iterator, and returns a measure that will produce the *next* state and
    // iterator. The iterator is passed by rvalue reference to support
    // move-only types like istream_view's iterator.
    const auto algo_l_step = [end](auto&& state, auto&& current_it) {
      using StateType = std::decay_t<decltype(state)>;
      using CurrentItType = std::decay_t<decltype(current_it)>;
      using NextPair = std::pair<StateType, CurrentItType>;
      // 1. Sample a uniform random number `u` in (0, 1).
      return uniform_real<double, Rng>(0.0, 1.0).and_then(
          [state = std::forward<decltype(state)>(state),
           current_it = std::forward<decltype(current_it)>(current_it),
           end](double u) mutable {
            // 2. Use `u` to calculate how many elements to skip. This is the
            //    core of Algorithm L's efficiency.
            const size_t skip_count = static_cast<size_t>(
                log(u) / log(1.0 - (1.0 / state.items_seen)));
            // 3. Advance the iterator by the calculated amount. We advance by
            //    `skip_count + 1` to land on the next replacement candidate.
            auto next_it = std::move(current_it);
            std::ranges::advance(next_it, skip_count + 1, end);
            // 4. If we've skipped past the end, fail the measure. This will
            //    terminate the gfold_m. Otherwise, create the new state.
            return guard<Rng>(next_it != end)
                .and_then([&](const std::monostate&) {
                  // The iterator is moved into the returned pair.
                  return pure<NextPair, Rng>(
                      {{*current_it, state.items_seen + skip_count + 1},
                       std::move(next_it)});
                });
          });
    };

    // Use the generalized fold to execute the sampling algorithm.
    // The initial state is the first element of the range. We must move the
    // iterator.
    return gfold_m(std::move(it), end, LoopState{*it, 1},
                   std::move(algo_l_step))
        // Once the fold is complete, extract the final reservoir item.
        .transform([](LoopState final_state) { return final_state.reservoir; });
  }
}

}  // namespace sub_prob_measures