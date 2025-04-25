import numpy as np
import pandas as pd

def generate_color_or_animal_data(n, animal_prop, hard_prop, misleading_text_length):
    colors = [
                "red", "blue", "green", "yellow", "orange",
                "purple", "pink", "brown", "black", "white",
                "cyan", "magenta", "lime", "teal", "indigo",
                "violet", "gold", "silver", "beige", "maroon"
            ]
    animals = [
                "lion", "tiger", "elephant", "giraffe", "zebra",
                "kangaroo", "panda", "koala", "dolphin", "whale",
                "eagle", "falcon", "bear", "wolf", "fox",
                "rabbit", "deer", "monkey", "hippopotamus", "rhinoceros"
            ]
    long_misleading_text='''Color theory is a conceptual framework used in visual arts, design, and many areas of visual communication to understand how colors relate to each other and how they can be combined to create pleasing or effective compositions. Rooted in both science and aesthetics, color theory explores the nature of color, the psychological impact it has on viewers, and the ways in which different colors interact. It informs countless decisions in fields ranging from painting and graphic design to interior decoration, fashion, marketing, and branding.

                    At the heart of color theory lies the color wheel, a circular diagram of colors arranged according to their chromatic relationship. The first known color wheel was developed by Sir Isaac Newton in the 17th century, who demonstrated that white light could be split into a spectrum of colors and then recombined into white light. His color circle laid the groundwork for modern color theory.

                    The traditional color wheel consists of three primary colors: red, yellow, and blue. These are the building blocks of all other colors, as they cannot be made by mixing any other colors together. By combining two primary colors, you get secondary colors: green, orange, and purple. Mixing a primary color with a neighboring secondary color produces tertiary colors such as red-orange or blue-green. These twelve hues form the basis of the artist’s color wheel.

                    Understanding how colors relate to one another on the wheel allows artists and designers to create color harmonies. Color harmony refers to aesthetically pleasing combinations of colors that evoke a sense of balance and unity. Some common types of color harmonies include complementary, analogous, triadic, and split-complementary.

                    Complementary colors are those located directly opposite each other on the color wheel, such as blue and orange or red and green. These pairs produce high contrast and high visual energy when used together, often making elements stand out sharply. Analogous colors are found next to each other on the wheel, such as blue, blue-green, and green. They share a similar hue and tend to be harmonious and soothing, often found in natural environments.

                    Triadic color schemes involve three colors that are evenly spaced around the color wheel, forming a triangle. An example of this would be red, yellow, and blue. This approach offers strong visual contrast while retaining balance and richness. Split-complementary schemes use a base color and the two colors adjacent to its complementary color. This creates a vibrant yet less jarring contrast than a direct complementary scheme.

                    Beyond hue relationships, color theory also takes into account other dimensions of color, such as value, saturation, and temperature. Value refers to the lightness or darkness of a color. For example, adding white to a color creates a tint, while adding black produces a shade. Saturation, or chroma, describes the intensity or purity of a color. Highly saturated colors appear vivid and intense, while desaturated colors appear more muted or gray.

                    Color temperature refers to the psychological association of colors with warmth or coolness. Warm colors such as red, orange, and yellow tend to evoke energy, warmth, and excitement. Cool colors like blue, green, and violet convey calmness, tranquility, and sometimes sadness. These associations are not just aesthetic—they have psychological and emotional impacts on viewers, which makes color choice critical in communication and design.'''
    long_misleading_text = long_misleading_text[:misleading_text_length]
    np.random.seed(0)
    data = {'id':[], 'value':[],'is_animal':[], 'animal_name':[]}
    for i in range(n):
        data['id'].append(i)
        is_animal = np.random.rand()<=animal_prop
        is_hard = np.random.rand()<=hard_prop
        if is_animal:
            val = np.random.choice(animals)
            data['is_animal'].append(True)
            data['animal_name'].append(val)
        else:
            val = np.random.choice(colors)
            data['is_animal'].append(False)
            data['animal_name'].append("")
        if is_hard:
            val = long_misleading_text[:len(long_misleading_text)//2] + f" {val} "+ long_misleading_text[len(long_misleading_text)//2:]
        data['value'].append(val)
    return pd.DataFrame.from_dict(data)