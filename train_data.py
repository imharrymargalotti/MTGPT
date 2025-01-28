import json
import random

# Define components for diverse query generation
colors = ["white", "blue", "black", "red", "green", "colorless", "multi"]
types = ["creature", "instant", "sorcery", "artifact", "enchantment", "planeswalker", "land"]
keywords = [
    "flying", "trample", "deathtouch", "haste", "lifelink", "flash", "hexproof",
    "first strike", "double strike", "vigilance", "reach", "indestructible", "cycling"
]
actions = [
    "show me", "find all", "list every", "get all", "search for", "display all"
]
effects = [
    "deals damage", "draws cards", "destroys target creature", "counters a spell",
    "adds mana", "creates tokens", "gains life", "exiles a card", "returns a card to your hand"
]

# Generate diverse examples
def generate_samples(sample_count):
    samples = set()
    while len(samples) < sample_count:
        action = random.choice(actions)
        color = random.choice(colors)
        card_type = random.choice(types)
        keyword = random.choice(keywords)
        effect = random.choice(effects)

        # Natural language query
        nl_query = f"{action} {color} {card_type}s with {keyword} that {effect}"

        # Scryfall query
        sf_query = f"type:{card_type}+color:{color[0]}+keyword:{keyword}+o:\"{effect}\""

        # Add as a tuple to avoid duplicates
        samples.add((nl_query, sf_query))

    # Convert to a list of lists for T5 model training
    return [[nl, sf] for nl, sf in samples]

# Generate the dataset
print("Generating dataset...")
dataset = generate_samples(5000)

# Save the dataset to a JSON file
output_file = "training_data.json"
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset generated and saved to {output_file}")
