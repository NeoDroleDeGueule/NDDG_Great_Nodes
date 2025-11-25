<b>NDDG Great Nodes</b>

________________________________________
________________________________________

<b>🍄Great Conditioning Modifier</b>


![GreatConditioningModifier](https://github.com/user-attachments/assets/e849c3aa-f770-45c7-9544-6259a66c1aa1)




  <b>📚 Modifier Guide</b>

🔹 > degree of importance for POSITIVE value modifications

🔸 > degree of importance for NEGATIVE value modifications

❌ > no use in Positive

<b>🔸 semantic_drift 🔹</b>

Progressive semantic drift
This modifier gradually blends your original prompt with a noisier version of itself, as if adding artistic blur to your instructions. With positive values, the image gently drifts away from the initial prompt while keeping overall coherence — imagine a concept "drifting" into neighboring interpretations. With negative values, the opposite occurs: the prompt is reinforced and becomes less prone to variation. Perfect for achieving creative variations without losing the original meaning.

<b>🔸🔸🔸 token_dropout 🔹🔹</b> 
<i>Does not work with Flux</i>

Selective token removal
Imagine your prompt is composed of several keywords the model "listens to." This modifier randomly ignores some of them, as if you temporarily changed the subject mid-generation. With positive values, some elements of your description are skipped, creating more abstract or surprising images because the model must "guess" the missing parts. With negative values, the opposite effect forces the model to concentrate on only a few specific tokens, producing cleaner, more focused images.

<b>🔸🔸🔸 gradient_amplify 🔹🔹</b>

Amplification of conceptual transitions
This modifier acts on the "transitions" between different elements of your prompt. Think of it as a contrast control for concepts: with positive values, the differences between parts of your description are exaggerated, creating more dramatic images with sharper contrasts between elements. With negative values, transitions are smoothed out, resulting in more harmonious, blended images where everything merges gently. Useful for controlling the dramatic intensity of your generations.

<b>🔸🔸🔸 guided_noise 🔹🔹🔹</b>

Proportional guided noise
This is the most universal and predictable modifier. It adds "creative noise" proportional to the intensity of your prompt — like adding film grain to a photo. With positive values (0.2–0.5), you get natural variations of your base image, perfect for generating several similar but unique versions. With negative values, you subtract this noise, stabilizing the image and making it more predictable. It's the ideal starting tool because its effects are progressive and controllable.

<b>🔸 quantize 🔹🔹🔹🔹</b>

Quantization and stabilization
This modifier reduces the “precision” of the instructions given to the model, like switching from millions of colors to a limited palette. With high positive values (0.5–1.0), the image becomes more stylized and graphic, with stronger choices and fewer subtle nuances — ideal for a simplified artistic rendering. With negative values, the opposite effect adds dithering (fine grain) that enriches details and micro-variations, creating more organic and textured images.

<b>🔸🔸🔸 perlin_noise 🔹🔹🔹🔹</b>

Coherent structured noise
Unlike classic random noise, Perlin noise creates smooth, “natural” variations, like cloud patterns or wood grain. With positive values, your images gain an organic, flowing quality, with soft variations that look natural rather than chaotic. Elements transform gradually instead of changing abruptly. With negative values, you get the opposite effect, which “de-structures” these patterns, creating more fragmented images. Excellent for natural or fluid abstract renderings.

<b>🔸🔸🔸 fourier_filter ❌</b>

NON-FUNCTIONAL frequency filtering
This modifier analyzes your prompt like a sound wave and filters certain conceptual “frequencies.” It only works with negative values: it's a low-pass filter that smooths the image by keeping only large shapes and general concepts (like keeping only bass tones). Think of it as an equalizer for your visual concepts.

<b>🔸 style_shift 🔹</b>

Directional style shift
This modifier pushes your prompt in a random but coherent “direction” in concept space, like turning a knob that gradually changes the global style. With positive values, you explore significant stylistic variations while keeping the subject — the image may shift from photorealistic to painterly, or from one lighting style to another. With negative values, the direction is reversed. Perfect for discovering unexpected stylistic interpretations of your prompt.

<b>🔸 temperature_scale 🔹</b>

Creativity control
This modifier controls the model’s “creative freedom,” exactly like the temperature parameter in text-based AIs. With positive values (0.5–1.0), the model becomes bolder and more unpredictable, taking artistic liberties with your prompt — ideal for creative exploration. With negative values, the model becomes conservative and predictable, following your prompt strictly with few variations — perfect for consistency and replication. It's the slider between “surprise me” and “do exactly what I say.”

<b>🔸 embedding_mix 🔹</b>
<i>Does not work with Flux</i>

Mixing and reorganization
This modifier rearranges the internal order of elements in your prompt, like shuffling a deck of cards. With positive values, different parts of your description are “mixed,” creating unexpected combinations — a character might inherit attributes intended for the background. With negative values, the effect “unmixes” by accentuating separations, making each element more distinct. Useful for creative hybridizations or, on the contrary, clearly separating concepts.

<b>🔸 svd_filter 🔹</b>

Complexity-based filtering (Advanced)
This modifier mathematically decomposes your prompt into “complexity components” and selectively modifies them. With positive values, it amplifies mid-level details, enriching nuances and visual sophistication. With negative values, it simplifies the concept by reducing those components, producing more minimalistic, clean images. Think of it as a filter that controls the “conceptual richness” of your generation.

<b>🔸 spherical_rotation 🔹</b>

Conceptual rotation (Advanced)
This modifier “rotates” your prompt in the multidimensional concept space while preserving its overall intensity, like rotating a 3D object. With high positive values, you get radical variations that keep the “weight” of the original prompt but explore entirely different angles. Results can be very surprising because the subject remains, but its interpretation changes dramatically. Excellent for extreme creative exploration.

<b>🔸 principal_component 🔹</b>

Modification of principal axes (Advanced)
This modifier identifies the “principal axes” of your prompt (the most important directions of variation) and alters them. With positive values, it amplifies these dominant axes, pushing the main features of your description to the extreme. With negative values, it attenuates them, simplifying the image by reducing conceptual dimensionality. It’s like choosing between “emphasize what matters most” and “flatten to simplify.”

<b>🔸 block_shuffle 🔹</b>

Block-based reorganization
This modifier cuts your prompt into conceptual “blocks” and rearranges them randomly while preserving coherence inside each block. With increasing positive values, the blocks become smaller and the shuffle more chaotic, creating surreal images where elements appear in unexpected order. It’s less radical than embedding_mix because local structure is preserved. Perfect for creating unusual compositions while keeping recognizable elements.

<b>💡 General Usage Tips</b>

• Beginners: Start with guided_noise (0.2–0.4) and temperature_scale (0.5–0.7)
• Subtle variations: perlin_noise (0.1–0.3), semantic_drift (0.2)
• Creative exploration: style_shift (0.5–0.8), spherical_rotation (0.6–1.0)
• Stabilization: Negative values on temperature_scale (–0.3 to –0.5)
• Artistic effects: quantize (0.7–1.0), block_shuffle (0.5–0.8)

<b>Don't forget: Change the seed of the node to get different variations with the same parameters!</b>



 
<img width="2310" height="900" alt="🍄Great_Conditioning_node" src="https://github.com/user-attachments/assets/1dbc3b63-c14e-49bb-b3ff-c5c2cd0f68c0" />

________________________________________
________________________________________

<b>🍄Great Interactive Gradient Node</b>
![Interactive_Gradient_Node](https://github.com/user-attachments/assets/94572120-eef0-496e-9b32-6506d0a68c2d)

________________________________________
________________________________________

<b>🍄Great Random Organic Gradient Node</b>

<img width="1194" height="816" alt="image" src="https://github.com/user-attachments/assets/f857fb4a-0fae-46a5-8540-2d34324e0b6e" />

________________________________________
________________________________________

<b>🍄Great_thick_border.js</b>

To see immediately which node is currently running!!

<img width="1423" height="777" alt="image" src="https://github.com/user-attachments/assets/5cb387c3-a447-419c-94e9-d52dc59ca197" />

<img width="361" height="779" alt="image" src="https://github.com/user-attachments/assets/36510418-e63a-4435-94b0-7a1504b12365" />

