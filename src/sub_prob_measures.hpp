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
auto bernoulli(const double p);

using RngDefault = std::mt19937;

/**
 * @brief Represents sub-probability measures as a failable, context-dependent
 * computation.
 *
 * @details This class is conditionally copyable. If it contains a simple,
 * copyable sampler, the object can be copied. If it contains a stateful,
 * move-only sampler (e.g., from a single-pass range), the object becomes
 * move-only.
 * This class is modelled after the `MaybeT Reader` monad transformer stack.
 * It encapsulates a function that depends on a shared environment (the random
 * number generator `Rng`) which:
 * 1. Can be any (likely stateful) function. Note that for a seeded random
 * number generator, the computation is deterministic.
 * Technically, this is more like a MaybeT State than a MaybeT Reader, but
 * we're abusing the fact that we're working in an imperative language, where
 * the stateful function can mutate its internal state.
 * 2. May fail to produce a value (`std::optional`).
 * The wrapped function signature is `Rng& -> std::optional<A>`.
 *
 * @tparam A The type of the value to be sampled.
 * @tparam Rng The type of the random number generator (e.g., std::mt19937).
 */
template <typename A, typename SamplerFunc, typename Rng = RngDefault>
class SubProbMeasure {
 private:
  SamplerFunc _sampler;

 public:
  /// The type of the value produced by the measure.
  using value_type = A;

  /**
   * @brief Constructs a SubProbMeasure from a callable sampler.
   * @param sampler The callable object that implements the sampling logic.
   */
  constexpr explicit SubProbMeasure(SamplerFunc&& sampler)
      : _sampler(std::move(sampler)) {}

  // --- Default Move Semantics ---
  // SubProbMeasure(const SubProbMeasure&) = default;
  // SubProbMeasure& operator=(const SubProbMeasure&) = default;
  // SubProbMeasure(SubProbMeasure&&) = default;
  // SubProbMeasure& operator=(SubProbMeasure&&) = default;

  /**
   * @brief Executes the probabilistic computation with a provided RNG.
   * @param rng A reference to the random number generator.
   * @return An optional containing the sampled value, or nullopt on failure.
   */
  constexpr std::optional<A> operator()(Rng& rng) noexcept(
      noexcept(this->_sampler(rng))) {
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
  std::optional<A> operator()() {
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
  constexpr auto and_then(F&& f) && {
    using B = typename decltype(f(std::declval<A>()))::value_type;

    auto new_sampler =
        [sampler = std::move(this->_sampler),
         f = std::forward<F>(f)](Rng& rng) mutable -> std::optional<B> {
      return sampler(rng).and_then(
          [&](const A a) { return f(std::move(a))(rng); });
    };
    return SubProbMeasure<B, decltype(new_sampler), Rng>(
        std::move(new_sampler));
  }

  /**
   * @brief Overload for lvalues (if copyable). Preserves the original object.
   */
  template <typename F>
  constexpr auto and_then(F&& f) const&
    requires std::is_copy_constructible_v<SamplerFunc>
  {
    return SubProbMeasure(*this).and_then(std::forward<F>(f));
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
  constexpr auto transform(F&& f) && {
    using B = decltype(f(std::declval<A>()));

    auto new_sampler = [sampler = this->_sampler, f = std::forward<F>(f)](
                                 Rng& rng) mutable -> std::optional<B> {
      // Run the original measure and then transform its optional result.
      return sampler(rng).transform(f);
    };

    return SubProbMeasure<B, decltype(new_sampler), Rng>(
        std::move(new_sampler));
  }

  template <typename F>
  constexpr auto transform(F&& f) const&
    requires std::is_copy_constructible_v<SamplerFunc>
  {
    return SubProbMeasure(*this).transform(std::forward<F>(f));
  }

  /**
   * @brief Returns a new measure by scaling this measure's probability.
   * @param factor The scaling factor in the interval [0, 1].
   * @return A new, scaled SubProbMeasure.
   */
  auto scale(const double factor) && {
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

  auto scale(const double factor) const&
    requires std::is_copy_constructible_v<SamplerFunc>
  {
    return SubProbMeasure(*this).scale(factor);
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
auto bernoulli(const double p) {
  // Get the RNG, then transform it into a boolean result by applying the
  // Bernoulli trial logic.
  return get_rng<Rng>().transform(
      [p](std::reference_wrapper<Rng> rng_ref) -> bool {
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
concept NumericDist =
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
  requires NumericDist<Dist, NumType, Rng>
auto numeric_dist(const NumType min, const NumType max) {
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
auto uniform_int(const IntType min, const IntType max) {
  return numeric_dist<std::uniform_int_distribution, IntType, Rng>(min, max);
}

/**
 * @brief Creates a measure for a uniform real distribution.
 * @param min The inclusive lower bound of the range.
 * @param max The exclusive upper bound of the range.
 * @return A measure that produces a `RealType`.
 */
template <std::floating_point RealType = double, typename Rng = RngDefault>
auto uniform_real(const RealType min, const RealType max) {
  return numeric_dist<std::uniform_real_distribution, RealType, Rng>(min, max);
}

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
auto uniform_range(Range&& range) {
  using T = std::ranges::range_value_t<Range>;

  if constexpr (std::ranges::forward_range<Range>) {
    // More efficient path for forward ranges (multi-pass capable).
    const auto is_empty = std::ranges::empty(range);
    return guard<Rng>(!is_empty).and_then(
        [r = std::forward<Range>(range)](const std::monostate&) {
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
    auto sampler = [range = std::forward<Range>(range)](
                       Rng& rng) mutable -> std::optional<T> {
      auto it = std::ranges::begin(range);
      const auto end = std::ranges::end(range);

      if (it == end) return std::nullopt;

      T initial_reservoir = *it;
      ++it;

      // Monadically generate initial W value.
      return uniform_real<double>(0.0, 1.0)(rng).and_then(
          [&](double u) -> std::optional<T> {
            // Simplified calculation for k = 1
            double w = std::exp(std::log(u));

            const auto loop = [&](this auto&& self, T current_reservoir,
                                  double current_w) -> std::optional<T> {
              // Monadically calculate how many elements to skip.
              return uniform_real<double>(0.0, 1.0)(rng).and_then(
                  [&](double skip_u) -> std::optional<T> {
                    const auto num_to_skip = static_cast<long>(std::floor(
                        std::log(skip_u) / std::log(1.0 - current_w)));
                    std::ranges::advance(it, num_to_skip, end);

                    if (it == end) return std::move(current_reservoir);

                    T next_reservoir = *it;
                    ++it;

                    // Monadically update W for the next iteration and
                    // recurse.
                    return uniform_real<double>(0.0, 1.0)(rng).and_then(
                        [&](double next_u) -> std::optional<T> {
                          double next_w =
                              current_w * std::exp(std::log(next_u));
                          return self(std::move(next_reservoir), next_w);
                        });
                  });
            };

            return loop(std::move(initial_reservoir), w);
          });
    };
    return SubProbMeasure<T, decltype(sampler), Rng>(std::move(sampler));
  }
}

/**
 * @brief Performs a monadic left fold over a range.
 *
 * This function iterates through a range, applying a monadic function `f` at
 * each step. The function `f` takes the current accumulator and the current
 * element, and returns a new `SubProbMeasure` containing the updated
 * accumulator.
 *
 * The implementation uses a functional, tail-recursive lambda (using C++23's
 * `deducing this`) to model the iteration. It uses `std::optional::and_then`
 * to propagate failure monadically, avoiding explicit checks and maintaining a
 * functional style. The structure is designed to allow for tail-call
 * optimization by compilers.
 *
 * @tparam Range The type of the input range.
 * @tparam B The type of the accumulator.
 * @tparam F The type of the monadic function.
 * @tparam Rng The type of the random number generator.
 * @param init The initial value of the accumulator.
 * @param range The range to fold over.
 * @param f The monadic function of type `(B, A) -> SubProbMeasure<B, Rng>`.
 * @return A `SubProbMeasure` representing the entire fold operation.
 */
template <std::ranges::input_range Range, typename B, typename F,
          typename Rng = RngDefault>
auto foldl_m(B init, Range&& range, F&& f) noexcept(
    std::is_nothrow_invocable_v<F, B, std::ranges::range_reference_t<Range>>) {
  // Helper to check if the folding function `f` is noexcept.
  static constexpr bool is_f_nothrow =
      noexcept(f(std::declval<B>(),
                 std::declval<std::ranges::range_reference_t<Range>>()));

  auto sampler =
      [init = std::move(init), range = std::forward<Range>(range),
       f = std::forward<F>(f)](
          Rng& rng) mutable noexcept(is_f_nothrow) -> std::optional<B> {
    auto it = std::ranges::begin(range);
    const auto end = std::ranges::end(range);

    const auto loop = [&](this auto&& self, std::optional<B> acc) noexcept(
                          is_f_nothrow) -> std::optional<B> {
      // Base case: Terminate if the range is exhausted.
      if (it == end) return acc;
      // Monadically compute the next accumulator state.
      // If 'acc' is nullopt, 'and_then' short-circuits and propagates it.
      const auto next_acc =
          acc.and_then([&](B next_acc_) noexcept(is_f_nothrow) {
            return f(std::move(next_acc_), *it)(rng);
          });
      ++it;
      return self(std::move(next_acc));
    };
    return loop(std::optional<B>(std::move(init)));
  };
  return SubProbMeasure<B, decltype(sampler), Rng>(std::move(sampler));
}

}  // namespace sub_prob_measures