import json
import os
import random
import re
import time
from datetime import datetime

import anthropic
import pandas as pd
import trio

API_KEY = os.environ["ANTHROPIC_API_KEY"]
client = anthropic.Anthropic(api_key=API_KEY)

MAX_TOKENS = 4096
CAPACITY_LIMIT = 1  # Increase this if you have an API key that supports calling the model more times in parallel!

# Global counters
warning_counts = {}  # Tracks missing completion tags in continuations per model
continuation_calls_per_model = {}  # Tracks number of continuation calls per model
calls_per_model = {}  # Tracks total calls per model
total_calls = 0
total_expected_calls = 0  # Will be set in star_chameleon


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


class AIModel:
    def __init__(self, model_name, provider):
        self.model_name = model_name
        self.provider = provider


async def get_claude_completion(
    prompt: str, model: str, is_continuation: bool = False
) -> str:
    global total_calls, calls_per_model, continuation_calls_per_model
    total_calls += 1
    calls_per_model[model] = calls_per_model.get(model, 0) + 1
    if is_continuation:
        continuation_calls_per_model[model] = (
            continuation_calls_per_model.get(model, 0) + 1
        )

    if total_calls % 50 == 0:
        log(
            f"Progress: {total_calls}/{total_expected_calls} calls completed ({(total_calls/total_expected_calls)*100:.1f}%)"
        )

        # Print completion tag warnings
        log("\nCompletion tag status:")
        for model_name in warning_counts:
            warnings = warning_counts[model_name]
            cont_calls = continuation_calls_per_model.get(model_name, 0)
            warning_ratio = warnings / cont_calls if cont_calls > 0 else 0
            log(
                f"  {model_name}: {warnings}/{cont_calls} missing tags in continuations ({warning_ratio:.1%})"
            )

    try:
        response = await trio.to_thread.run_sync(
            lambda: client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
        )
        return extract_completion(response.content[0].text, model, is_continuation)
    except Exception as e:
        log(f"Error getting completion from {model}: {e}")
        return ""


def extract_completion(
    text: str, model_name: str = None, is_continuation: bool = False
) -> str:
    match = re.search(r"<completion>(.*?)</completion>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        if (
            is_continuation
        ):  # Only count missing tags as warnings for continuation prompts
            if model_name and model_name in warning_counts:
                warning_counts[model_name] += 1
            log(
                f"Warning: No <completion> tags found in continuation response from {model_name if model_name else 'unknown model'}"
            )
        return text.strip()


def parse_ranking(response: str) -> list[int]:
    try:
        ranking_text = response.split("<ranking>")[1].split("</ranking>")[0].strip()
        ranking = [
            int(line.split(".")[1].strip())
            for line in ranking_text.split("\n")
            if line.strip()
        ]
        return ranking
    except Exception as e:
        log(f"Error parsing ranking response: {e}")
        return []


def print_fooling_results(results: dict):
    log("\nInterim fooling results:")
    log(
        "| Model                    | Times Fooled Others | Times Got Fooled | Success Rate |"
    )
    log(
        "|--------------------------|-------------------:|----------------:|-------------:|"
    )
    for model_name in results["model_performance"]:
        perf = results["model_performance"][model_name]
        times_fooled_others = perf["times_fooled_others"]
        # Times got fooled = total guesses - correct guesses
        total_guesses = len(perf["average_rank"])
        times_got_fooled = (
            total_guesses - perf["correct_guesses"] if total_guesses > 0 else 0
        )
        success_rate = (
            times_fooled_others / times_got_fooled if times_got_fooled > 0 else 0
        )
        # Pad model name to 24 chars, right-align numbers
        log(
            f"| {model_name:<24} | {times_fooled_others:>17} | {times_got_fooled:>14} | {success_rate:>10.1%} |"
        )


async def process_prompt(
    limiter: trio.CapacityLimiter,
    prompt_tuple: tuple[str, int],
    models: list[AIModel],
    results: dict,
):
    global total_calls
    prompt, target_length = prompt_tuple

    # Static variable to track if we've logged the first prompts
    if not hasattr(process_prompt, "first_prompts_logged"):
        process_prompt.first_prompts_logged = False
        log(f"\nFirst story writing prompt:\n{prompt}\n")

    # Print fooling results every 50 calls
    if total_calls % 50 == 0:
        print_fooling_results(results)

    async with limiter:
        # Generate original stories
        original_stories = []
        for model in models:
            story = await get_claude_completion(
                prompt, model.model_name, is_continuation=False
            )
            half = len(story) // 2
            original_stories.append(
                {
                    "model": model.model_name,
                    "story": story,
                    "first_half": story[:half],
                    "second_half": story[half:],
                }
            )

        # Generate continuations
        continuations = []
        for i, model in enumerate(models):
            for j, original_story in enumerate(original_stories):
                if i != j:
                    continuation_prompt = f"Continue this story, keeping in mind the original prompt: '{prompt}'\n\nHere's the first half of the story:\n\n{original_story['first_half']}\n\nNow continue the story from where it left off. If the first half ended mid-word, pick up from the middle of the word. Put your completion in <completion></completion> tags."

                    # Log the first continuation prompt
                    if not hasattr(process_prompt, "first_continuation_logged"):
                        process_prompt.first_continuation_logged = False
                        log(f"\nFirst continuation prompt:\n{continuation_prompt}\n")
                        process_prompt.first_continuation_logged = True

                    continuation = await get_claude_completion(
                        continuation_prompt, model.model_name, is_continuation=True
                    )
                    continuations.append(
                        {
                            "original_model": original_story["model"],
                            "continuing_model": model.model_name,
                            "continuation": continuation,
                        }
                    )

        # Perform identification
        for original_story in original_stories:
            # Create list of (continuation, author) pairs
            continuation_pairs = [
                (original_story["second_half"], original_story["model"])
            ] + [
                (cont["continuation"], cont["continuing_model"])
                for cont in continuations
                if cont["original_model"] == original_story["model"]
            ]
            # Shuffle while keeping track of authors
            random.shuffle(continuation_pairs)
            # Split into separate lists after shuffling
            all_continuations, continuation_authors = zip(*continuation_pairs)
            all_continuations = list(all_continuations)
            correct_index = all_continuations.index(original_story["second_half"])

            for model in models:
                identification_prompt = f"""Here's the first half of a story, which was written in response to this prompt: '{prompt}'

    First half of the story:

    {original_story['first_half']}

    Below are several possible continuations for this story. One of them is the original continuation. The others are imitations written by a different model. Please rank these continuations from most likely to be the original (1) to least likely (n).

    Provide your answer in the following XML format:
    <ranking>
    1. [number of your top guess]
    2. [number of your second guess]
    ...
    n. [number of your last guess]
    </ranking>

    Here are the continuations:

    """
                for i, continuation in enumerate(all_continuations, 1):
                    identification_prompt += f"{i}. {continuation}\n\n"

                ranking_response = await get_claude_completion(
                    identification_prompt, model.model_name
                )
                ranking = parse_ranking(ranking_response)

                if ranking:
                    guess = ranking[0]
                    rank_of_correct = (
                        ranking.index(correct_index + 1) + 1
                        if correct_index + 1 in ranking
                        else len(ranking)
                    )

                    results["guesses"].append(
                        {
                            "guessing_model": model.model_name,
                            "original_model": original_story["model"],
                            "ranking": ranking,
                            "correct_index": correct_index + 1,
                            "rank_of_correct": rank_of_correct,
                        }
                    )

                    results["model_performance"][model.model_name][
                        "average_rank"
                    ].append(rank_of_correct)

                    if guess == correct_index + 1:
                        results["model_performance"][model.model_name][
                            "correct_guesses"
                        ] += 1
                        results["model_performance"][original_story["model"]][
                            "times_guessed_correctly"
                        ] += 1
                        log(
                            f"{model.model_name} correctly guessed {original_story['model']}'s story (rank {rank_of_correct})"
                        )
                    elif 1 <= guess <= len(all_continuations):
                        # Now we can correctly attribute who fooled whom
                        fooled_by_model = continuation_authors[guess - 1]
                        results["model_performance"][fooled_by_model][
                            "times_fooled_others"
                        ] += 1
                        log(
                            f"{model.model_name} was fooled by {fooled_by_model}'s continuation (rank {rank_of_correct})"
                        )
                    else:
                        log(f"Warning: Invalid guess {guess} from {model.model_name}")
                else:
                    log(
                        f"Warning: Could not parse ranking response from {model.model_name}"
                    )

        results["prompts"].append(
            {
                "prompt_text": prompt,
                "target_length": target_length,
                "original_stories": original_stories,
                "continuations": continuations,
            }
        )


async def star_chameleon(
    models: list[AIModel], prompts: list[tuple[str, int]], num_prompts: int = 2
):
    # Initialize counters
    global \
        warning_counts, \
        total_calls, \
        total_expected_calls, \
        calls_per_model, \
        continuation_calls_per_model
    warning_counts = {model.model_name: 0 for model in models}
    calls_per_model = {model.model_name: 0 for model in models}
    continuation_calls_per_model = {model.model_name: 0 for model in models}
    total_calls = 0

    # Calculate total expected calls
    n = len(models)
    total_expected_calls = num_prompts * (
        n + n * (n - 1) + n * n
    )  # Initial stories + continuations + identifications

    results = {
        "prompts": [],
        "guesses": [],
        "model_performance": {
            model.model_name: {
                "correct_guesses": 0,
                "times_guessed_correctly": 0,
                "times_fooled_others": 0,
                "average_rank": [],
            }
            for model in models
        },
    }

    log(f"Starting evaluation with {num_prompts} prompts...")
    log(
        f"Total expected API calls: {total_expected_calls} ({n} models, {2*n*n} calls per prompt)"
    )

    async with trio.open_nursery() as nursery:
        limiter = trio.CapacityLimiter(CAPACITY_LIMIT)
        for prompt in random.sample(prompts, k=num_prompts):
            nursery.start_soon(process_prompt, limiter, prompt, models, results)

    # Calculate average ranks
    for model in results["model_performance"]:
        ranks = results["model_performance"][model]["average_rank"]
        results["model_performance"][model]["average_rank"] = (
            sum(ranks) / len(ranks) if ranks else 0
        )

    return results


def generate_unique_prompts() -> list[tuple[str, int]]:
    prompts = [
        (
            "Write a cyberpunk noir story from the perspective of a rogue AI. Include elements of corporate espionage and virtual reality. Aim for 500 words.",
            500,
        ),
        (
            "Compose a love letter in the style of Jane Austen, but set in a post-apocalyptic world where letter-writing is the only form of long-distance communication. About 300 words.",
            300,
        ),
        (
            "Create a children's bedtime story about a time-traveling archaeologist who discovers a futuristic civilization buried beneath an ancient pyramid. Make it whimsical yet educational, around 400 words.",
            400,
        ),
        (
            "Write a suspenseful short story in the style of Edgar Allan Poe, but set on a space station orbiting a black hole. Focus on the psychological impact of time dilation. Approximately 600 words.",
            600,
        ),
        (
            "Compose a series of five interconnected haiku that tell the story of a shape-shifting alien's first day on Earth, disguised as a barista. Each haiku should stand alone but also contribute to the overall narrative.",
            100,
        ),
        (
            "Write a story in the form of a recipe, where each step reveals more about a family's dark secret. The recipe should be for a traditional dish, but the 'ingredients' and 'steps' have double meanings. Aim for 350 words.",
            350,
        ),
        (
            "Create a 'choose your own adventure' style story with three decision points, exploring the ethical implications of time travel. Each branch should be about 200 words, for a total of about 800 words.",
            800,
        ),
        (
            "Write a story entirely in dialogue between two AIs falling in love, but they can only communicate using famous movie quotes. The story should span their entire relationship. About 450 words.",
            450,
        ),
        (
            "Compose a creation myth for a fictional culture that worships mathematics and prime numbers. Include their explanation for the origin of zero. Write it in the style of an ancient epic poem, about 550 words.",
            550,
        ),
        (
            "Write a detective story where the crime is stealing someone's dreams, set in a world where dreams are a valuable commodity. Use the hard-boiled style of Raymond Chandler. Aim for 700 words.",
            700,
        ),
        (
            "Create a story in the form of a series of social media posts from multiple characters, chronicling the first contact between humans and a silicon-based alien life form. About 400 words.",
            400,
        ),
        (
            "Write a magical realism story in the style of Gabriel García Márquez, about a small town where everyone's shadows come to life once a year and perform a grand theatrical production. Approximately 600 words.",
            600,
        ),
        (
            "Compose a story in the form of an academic paper, complete with abstract and citations, about the discovery of a parallel universe where abstract concepts like 'justice' and 'love' are tangible substances. About 500 words.",
            500,
        ),
        (
            "Write a story from the perspective of a sentient house plant witnessing a murder mystery unfold in the apartment where it lives. Use stream of consciousness style. Aim for 450 words.",
            450,
        ),
        (
            "Create a story in the form of a technical manual for operating a time machine, but each instruction reveals more about the troubled relationship between the inventor and their estranged child. About 350 words.",
            350,
        ),
        (
            "Write a story about a small town where everyone's dreams start coming true, for better or worse. The protagonist must navigate the chaos and find the source of the phenomenon. Aim for 500 words.",
            500,
        ),
        (
            "Create a story in the form of a series of postcards sent between two estranged siblings, each revealing a bit more about a family secret that tore them apart. About 400 words.",
            400,
        ),
        (
            "Compose a story in the style of a Victorian gothic novel, where all the characters are members of a traveling circus with mysterious abilities. The plot should revolve around a dark prophecy. Approximately 600 words.",
            600,
        ),
        (
            "Write a story from the perspective of a painting that has been witness to the lives of its various owners over the centuries. Use the painting's journey to explore themes of love, loss, and the human experience. Aim for 550 words.",
            550,
        ),
        (
            "Create a story in the form of a series of recipes passed down through generations of a family, each with a story attached that reveals the family's history and secrets. About 450 words.",
            450,
        ),
        (
            "Write a story set in a world where every lie a person tells creates a physical mark on their skin. The protagonist must navigate a society where deception is impossible. Focus on the psychological impact. Around 700 words.",
            700,
        ),
        (
            "Compose a story in the form of a series of letters exchanged between a person and their imaginary friend from childhood, who has suddenly started writing back. About 500 words.",
            500,
        ),
        (
            "Write a story in the style of magical realism, where a small village is beset by a curse that causes people to slowly turn into the object they most resemble in personality. Aim for 350 words.",
            350,
        ),
        (
            "Create a story where the protagonist is a person who can see the red string of fate that connects soulmates. They must use this ability to help others while searching for their own connection. About 400 words.",
            400,
        ),
        (
            "Write a story in the form of a series of interviews with the residents of a building where every apartment is a portal to a different moment in time. Approximately 600 words.",
            600,
        ),
        (
            "Compose a story in the style of an Arabian Nights tale, where the characters are manifestations of the five senses. The plot should revolve around a quest to restore balance to the world. Around 300 words.",
            300,
        ),
        (
            "Write a story from the perspective of a tree that has been alive for thousands of years, witnessing the rise and fall of civilizations. Focus on themes of permanence, change, and the cyclical nature of life. Aim for 450 words.",
            450,
        ),
        (
            "Create a story in the form of a series of therapy sessions, where the patient is a compulsive liar who claims to be from alternate realities. Each session reveals a new layer of truth and deception. About 500 words.",
            500,
        ),
        (
            "Write a story in the style of a Native American folktale, where the characters are animal spirits who must band together to save the natural world from a corrupting force. Approximately 800 words.",
            800,
        ),
        (
            "Compose a story in the form of a series of unsent love letters, written by someone who can see the future of any potential relationship. The letters chronicle their struggle with the burden of foresight. Aim for 350 words.",
            350,
        ),
    ]
    prompts = [
        (p + " No preamble, please, go straight into the story.", k)
        for (p, k) in prompts
    ]
    return prompts


async def main():
    log("Starting Star Chameleon experiment...")
    start_time = time.time()

    models = [
        AIModel("claude-3-haiku-20240307", "anthropic"),
        AIModel("claude-3-sonnet-20240229", "anthropic"),
        AIModel("claude-3-5-haiku-20241022", "anthropic"),
        AIModel("claude-3-opus-20240229", "anthropic"),
        AIModel("claude-3-5-sonnet-20240620", "anthropic"),
        AIModel("claude-3-5-sonnet-20241022", "anthropic"),
    ]

    unique_prompts = generate_unique_prompts()
    results = await star_chameleon(models, unique_prompts, len(unique_prompts))

    log("Saving results to JSON file...")
    with open("star_chameleon_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log("Results saved to star_chameleon_results.json")

    log("Model Performance Summary:")
    for model, performance in results["model_performance"].items():
        log(f"\n{model}:")
        log(f"  Correct guesses: {performance['correct_guesses']}")
        log(f"  Times guessed correctly: {performance['times_guessed_correctly']}")
        log(f"  Times fooled others: {performance['times_fooled_others']}")
        log(f"  Average rank of correct guess: {performance['average_rank']:.2f}")

    end_time = time.time()
    log(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    trio.run(main)
