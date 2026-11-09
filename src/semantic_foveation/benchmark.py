"""60-prompt small-object benchmark: wide scenes where one small object carries the meaning of the image."""

import csv
import os
from typing import Dict, List

_RAW = {
    "animal": [
        ("A wide shot of a quiet city park with green lawns and trees, a red fox sitting on a wooden bench on the right side", "a red fox"),
        ("A snowy pine forest clearing in winter, a small snowy owl perched on a fence post in the lower left", "a snowy owl"),
        ("A vast golden wheat field under a blue sky, a hare standing upright in the middle distance", "a hare"),
        ("A kitchen counter with pots, jars and a window, an orange tabby cat curled up on the windowsill", "an orange tabby cat"),
        ("A misty river valley at dawn, a grey heron standing in the shallow water near the far bank", "a grey heron"),
        ("A tropical beach with palm trees and turquoise water, a small sea turtle crawling on the sand in the foreground", "a sea turtle"),
        ("A garden with rows of colorful tulips, a single monarch butterfly resting on a yellow tulip", "a monarch butterfly"),
        ("A rocky mountain slope with scattered boulders, a mountain goat standing on a ledge near the top right", "a mountain goat"),
        ("A cozy living room with a sofa, rug and bookshelves, a small beagle puppy sleeping on the rug", "a beagle puppy"),
        ("An autumn forest floor covered in orange leaves, a red squirrel holding an acorn on a tree stump", "a red squirrel"),
        ("A calm lake surrounded by reeds at sunset, a white swan floating on the water in the distance", "a white swan"),
        ("A savanna with acacia trees and tall dry grass, a cheetah resting under a tree on the left", "a cheetah"),
        ("A coral reef underwater scene with blue water, a small clownfish peeking out of a sea anemone", "a clownfish"),
        ("A farmyard with a red barn and hay bales, a rooster standing on top of a wooden fence", "a rooster"),
        ("A desert landscape with sand dunes and a clear sky, a small fennec fox on top of a dune", "a fennec fox"),
    ],
    "object": [
        ("A cluttered wooden desk seen from above, with papers, books and coffee mugs, and a gold pocket watch with an engraved face near the corner", "a gold pocket watch"),
        ("A sunny picnic blanket on grass with fruit and sandwiches, a glass bottle of lemonade standing on the edge", "a glass bottle of lemonade"),
        ("A messy workshop bench with tools and wood shavings, a small brass compass lying among the tools", "a brass compass"),
        ("A wide view of a library reading room with tall shelves, an open red leather book on a table in the distance", "a red leather book"),
        ("A minimalist white bathroom, a small green succulent plant in a ceramic pot on the sink", "a succulent in a pot"),
        ("A rainy city street at night with reflections, a yellow umbrella lying open on the sidewalk", "a yellow umbrella"),
        ("A kitchen table with plates and cutlery, a single slice of strawberry cake on a small plate", "a slice of strawberry cake"),
        ("A sandy beach with footprints, a colorful seashell lying near the waterline", "a seashell"),
        ("A cozy bedroom with a bed and nightstand, a vintage alarm clock on the nightstand", "a vintage alarm clock"),
        ("A grassy football field seen from the stands, a white soccer ball near the goal post", "a soccer ball"),
        ("An office desk with a laptop, papers and pens, a small blue coffee mug with steam rising", "a blue coffee mug"),
        ("A forest trail covered in moss, a red mushroom with white spots growing beside a log", "a red mushroom with white spots"),
        ("A wooden dock on a calm lake, a classic film camera resting on the edge of the dock", "a film camera"),
        ("A snowy front porch with a wooden door, a small wrapped gift box with a red ribbon on the doormat", "a gift box with a red ribbon"),
        ("A bright art studio with canvases and easels, a paint palette with colorful paint blobs on a stool", "a paint palette"),
    ],
    "person": [
        ("A vast desert with towering red rock formations, a lone hiker with a blue backpack walking on the trail", "a hiker with a blue backpack"),
        ("A wide snowy mountain landscape, a skier in a red jacket descending the slope in the distance", "a skier in a red jacket"),
        ("A large empty city square with pigeons and old buildings, a street musician playing a violin near a fountain", "a violinist"),
        ("A misty pier stretching into the ocean, a fisherman in a yellow raincoat at the end of the pier", "a fisherman in a yellow raincoat"),
        ("A sunflower field at golden hour, a little girl in a white dress standing among the sunflowers", "a girl in a white dress"),
        ("A wide ocean view with big waves, a surfer riding a wave in the middle distance", "a surfer"),
        ("A large rooftop garden with plants and city skyline, an old man reading a newspaper on a bench", "an old man reading a newspaper"),
        ("A green valley with a winding river, a person paddling a red kayak on the river", "a person in a red kayak"),
        ("A quiet train station platform with benches, a woman in a green coat holding a suitcase", "a woman in a green coat"),
        ("A wide grassy hill under a cloudy sky, a boy flying a kite at the top of the hill", "a boy flying a kite"),
        ("A large kitchen in a restaurant with steel counters, a chef in a white hat plating a dish", "a chef in a white hat"),
        ("A frozen lake surrounded by forest, an ice skater in a purple outfit spinning on the ice", "an ice skater"),
        ("A busy farmers market street with stalls, a vendor holding a basket of red apples", "a vendor holding a basket of apples"),
        ("A rocky cliff face above the sea, a rock climber with a red helmet halfway up the cliff", "a rock climber"),
        ("A wide park with autumn trees, a jogger in a bright orange shirt running on the path", "a jogger in an orange shirt"),
    ],
    "structure": [
        ("A large empty beach at sunset with gentle waves, a small lighthouse on a rocky outcrop in the distance", "a lighthouse"),
        ("A vast green countryside with rolling hills, a small white chapel with a bell tower on a hill", "a white chapel"),
        ("A wide alpine lake surrounded by mountains, a small red wooden cabin on the shore", "a red wooden cabin"),
        ("A misty rice terrace landscape, a small wooden hut with a thatched roof among the terraces", "a thatched hut"),
        ("A wide prairie under a dramatic sky, an old windmill standing alone in the field", "an old windmill"),
        ("A calm harbor with water and hills, a small red sailboat moored near the dock", "a red sailboat"),
        ("A snowy mountain pass, a yellow cable car hanging from a wire in the middle of the image", "a yellow cable car"),
        ("A long country road through autumn trees, a red vintage car parked on the side of the road", "a red vintage car"),
        ("A wide desert highway at dusk, a small neon-lit diner on the side of the road", "a neon diner"),
        ("A dense green jungle canopy seen from above, a small stone temple ruin peeking through the trees", "a stone temple ruin"),
        ("A frozen tundra landscape, a small orange tent pitched near a frozen lake", "an orange tent"),
        ("A wide river with forested banks, a small wooden footbridge crossing the river in the distance", "a wooden footbridge"),
        ("A large green field at dusk, a hot air balloon with rainbow stripes floating low above the field", "a hot air balloon"),
        ("A wide sandy beach, a colorful lifeguard tower standing near the water", "a lifeguard tower"),
        ("A quiet railway through a green valley, a small red steam train crossing a stone viaduct", "a red steam train"),
    ],
}

BENCHMARK_PROMPTS: List[Dict] = [
    {"id": i, "category": cat, "prompt": prompt, "focus_prompt": focus, "seed": 100 + i}
    for i, (cat, prompt, focus) in enumerate(
        (cat, prompt, focus) for cat, items in _RAW.items() for prompt, focus in items
    )
]


def save_benchmark_csv(out_csv: str = "data/small_object_benchmark_60.csv") -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "prompt", "focus_prompt", "seed"])
        writer.writeheader()
        writer.writerows(BENCHMARK_PROMPTS)


if __name__ == "__main__":
    save_benchmark_csv()
    print(f"Saved {len(BENCHMARK_PROMPTS)} prompts")
