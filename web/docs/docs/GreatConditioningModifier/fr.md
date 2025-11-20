# Great Conditiong Modifier

💡 Conseils généraux d'utilisation</br>

• Débutants : commencez avec guided_noise (0,2--0,4) et
temperature_scale (0,5--0,7)</br>
• Variations subtiles : perlin_noise (0,1--0,3), semantic_drift (0,2)</br>
• Exploration créative : style_shift (0,5--0,8), spherical_rotation
(0,6--1,0)</br>
• Stabilisation : valeurs négatives sur temperature_scale (--0,3 à
--0,5)</br>
• Effets artistiques : quantize (0,7--1,0), block_shuffle (0,5--0,8)

N'oubliez pas : changez le seed du nœud pour obtenir différentes
variations avec les mêmes paramètres !

📚 Guide des Modificateurs

🔹 </br>> degré d'importance pour les valeurs POSITIVES</br>
🔸 </br>> degré d'importance pour les valeurs NÉGATIVES</br>
❌ </br>> aucune utilité en positif

🔸 semantic_drift 🔹
Dérive sémantique progressive</br>
Ce modificateur mélange progressivement votre prompt original avec une
version plus bruitée, comme si vous ajoutiez un flou artistique à vos
instructions. Avec des valeurs positives, l'image s'éloigne doucement du
prompt initial tout en conservant sa cohérence globale --- imaginez un
concept qui « dérive » vers des interprétations voisines. Avec des
valeurs négatives, l'effet inverse renforce le prompt et réduit la
variabilité. Parfait pour obtenir des variations créatives sans perdre
le sens central.

🔸🔸🔸 token_dropout 🔹🔹 *(ne fonctionne pas avec Flux)*
Suppression sélective de tokens</br>
Ce modificateur ignore aléatoirement certaines parties de votre prompt,
comme si vous changiez brièvement de sujet. Avec des valeurs positives,
certaines informations sont omises, produisant des images plus
abstraites ou surprenantes. Avec des valeurs négatives, le modèle se
concentre davantage sur quelques tokens clés.

🔸🔸🔸 gradient_amplify 🔹🔹
Amplification des transitions conceptuelles</br>
Il agit comme un contrôle de contraste conceptuel : valeurs positives →
transitions accentuées et rendu dramatique ; valeurs négatives →
transitions adoucies et rendu harmonieux.

🔸🔸🔸 guided_noise 🔹🔹🔹
Bruit guidé proportionnel</br>
Ajoute un « bruit créatif » naturel comparable au grain d'une photo.
Valeurs positives (0,2--0,5) → variations naturelles du rendu. Valeurs
négatives → stabilisation et images plus prévisibles. C'est l'un des
modificateurs les plus fiables.

🔸 quantize 🔹🔹🔹🔹
Quantification et stabilisation</br>
Réduit la précision des instructions, comme passer d'un large spectre de
couleurs à une palette limitée. Valeurs positives (0,5--1,0) → rendu
stylisé et graphique. Valeurs négatives → ajout de dithering, détails
enrichis et textures organiques.

🔸🔸🔸 perlin_noise 🔹🔹🔹🔹
Bruit structuré cohérent</br>
Produit des variations organiques proches de motifs naturels (nuages,
bois, etc.). Positif → transformations progressives et naturelles.
Négatif → fragmentation des motifs.

🔸🔸🔸 fourier_filter ❌
Filtrage fréquentiel (non fonctionnel en positif)</br>
Agit comme un filtre passe-bas conceptuel : seules les grandes formes et
idées générales sont conservées.

🔸 style_shift 🔹
Changement directionnel de style</br>
Modifie de manière cohérente le style global tout en gardant le sujet.
Utile pour explorer divers rendus stylistiques.

🔸 temperature_scale 🔹
Contrôle de créativité</br>
Positif (0,5--1,0) → plus de liberté créative et surprises.</br>
Négatif → interprétation stricte et cohérente.

🔸 embedding_mix 🔹 *(ne fonctionne pas avec Flux)*
Mélange et réorganisation interne des concepts.

🔸 svd_filter 🔹
Filtrage basé sur la complexité</br>
Positif → enrichit les détails.</br>
Négatif → simplifie l'image.

🔸 spherical_rotation 🔹
Rotation conceptuelle (avancé)</br>
Conserve l'intensité du prompt mais change l'interprétation de manière
radicale.

🔸 principal_component 🔹
Modification des axes principaux du prompt.

🔸 block_shuffle 🔹
Réorganisation en blocs</br>
Crée des compositions inattendues tout en préservant la cohérence
locale.
