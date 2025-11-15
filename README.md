<b>NDDG Great Nodes</b>

________________________________________
________________________________________

<b>🍄Great Conditioning Modifier</b>

<img width="533" height="434" alt="image" src="https://github.com/user-attachments/assets/78f5e2c1-66fa-4486-aee0-4754942648e4" />

<b>📚 Guide des Modificateurs</b>

🔹 > degré d'importance des modifications des valeurs POSITIVES

🔸 > degré d'importance des modifications des valeurs NEGATIVES

❌ > pas d'utilisation en Positif

<b>🔸 semantic_drift 🔹</b>

Dérive sémantique progressive
Ce modificateur mélange progressivement votre prompt original avec une version bruitée de lui-même, comme si vous ajoutiez du flou artistique à vos instructions. Avec des valeurs positives, l'image s'éloigne doucement du prompt initial tout en gardant une cohérence globale - imaginez un concept qui "dérive" vers des interprétations voisines. Avec des valeurs négatives, l'effet inverse se produit : le prompt est renforcé et moins sujet à variation. Parfait pour obtenir des variations créatives sans perdre complètement le sens original.
________________________________________

<b>🔸🔸🔸 token_dropout 🔹🔹</b>

Suppression sélective de tokens
Imagine que votre prompt soit composé de plusieurs mots-clés que le modèle "écoute". Ce modificateur en ignore aléatoirement certains, comme si vous changiez temporairement de sujet en cours de génération. Avec des valeurs positives, certains éléments de votre description sont ignorés, créant des images plus abstraites ou surprenantes car le modèle doit "deviner" les parties manquantes. Avec des valeurs négatives, l'effet inverse force le modèle à se concentrer uniquement sur quelques tokens spécifiques, créant des images plus épurées et focalisées.
________________________________________

<b>🔸🔸🔸 gradient_amplify 🔹🔹</b>

Amplification des transitions conceptuelles
Ce modificateur agit sur les "transitions" entre les différents éléments de votre prompt. Pensez-y comme un contrôle de contraste pour les concepts : avec des valeurs positives, les différences entre les parties de votre description sont exagérées, créant des images plus dramatiques avec des contrastes marqués entre les éléments. Avec des valeurs négatives, les transitions sont lissées, donnant des images plus harmonieuses et fondues, où tout se mélange en douceur. Utile pour contrôler l'intensité dramatique de vos générations.
________________________________________

<b>🔸🔸🔸 guided_noise 🔹🔹🔹</b>

Bruit guidé proportionnel
C'est le modificateur le plus universel et prévisible. Il ajoute du "bruit créatif" proportionnel à l'intensité de votre prompt - comme ajouter du grain à une photo. Avec des valeurs positives (0.2-0.5), vous obtenez des variations naturelles de votre image de base, parfait pour générer plusieurs versions similaires mais uniques. Avec des valeurs négatives, vous soustrayez ce bruit, stabilisant l'image et la rendant plus prévisible. C'est l'outil idéal pour commencer car ses effets sont progressifs et contrôlables.
________________________________________

<b>🔸 quantize 🔹🔹🔹🔹</b>

Quantification et stabilisation
Ce modificateur réduit la "précision" des instructions données au modèle, comme passer d'une image en millions de couleurs à une palette limitée. Avec des valeurs positives élevées (0.5-1.0), l'image devient plus stylisée et graphique, avec des choix plus tranchés et moins de nuances subtiles - idéal pour un rendu artistique simplifié. Avec des valeurs négatives, l'effet inverse ajoute du dithering (grain fin) qui enrichit les détails et les micro-variations, créant des images plus organiques et texturées.
________________________________________

<b>🔸🔸🔸 perlin_noise 🔹🔹🔹🔹</b>

Bruit structuré cohérent
Contrairement au bruit aléatoire classique, le bruit de Perlin crée des variations "naturelles" et continues, comme les motifs des nuages ou du bois. Avec des valeurs positives, vos images acquièrent une qualité organique fluide, avec des variations douces qui semblent naturelles plutôt que chaotiques. Les éléments se transforment progressivement au lieu de changer brusquement. Avec des valeurs négatives, vous obtenez l'effet inverse qui "dé-structure" ces patterns, créant des images plus fragmentées. Excellent pour des rendus naturels ou abstraits fluides.

________________________________________

<b>🔸🔸🔸 fourier_filter ❌</b>

Filtrage fréquentiel NON FONCTIONNEL
Ce modificateur analyse votre prompt comme une onde sonore et filtre certaines "fréquences" conceptuelles. Se s’utilise qu’avec des valeurs négatives, c'est un filtre passe-bas qui lisse l'image en gardant seulement les grandes formes et concepts généraux (comme garder uniquement les basses). Pensez-y comme un équaliseur pour vos concepts visuels.
________________________________________

<b>🔸 style_shift 🔹</b>

Décalage directionnel du style
Ce modificateur pousse votre prompt dans une "direction" aléatoire mais cohérente dans l'espace des concepts, comme tourner un bouton qui change progressivement le style global. Avec des valeurs positives, vous explorez des variations stylistiques importantes tout en gardant le sujet - l'image peut passer d'un style photoréaliste à pictural, ou d'un éclairage à un autre. Avec des valeurs négatives, la direction est inversée. Parfait pour découvrir des interprétations stylistiques inattendues de votre prompt.
________________________________________

<b>🔸 temperature_scale 🔹</b>

Contrôle de créativité
Ce modificateur contrôle la "liberté créative" du modèle, exactement comme le paramètre temperature des IA textuelles. Avec des valeurs positives (0.5-1.0), le modèle devient plus audacieux et imprévisible, prenant des libertés artistiques avec votre prompt - idéal pour l'exploration créative. Avec des valeurs négatives, le modèle devient conservateur et prévisible, suivant votre prompt à la lettre avec peu de variations - parfait pour la consistance et la reproduction. C'est le curseur entre "surprends-moi" et "fais exactement ce que je dis".
________________________________________

<b>🔸 embedding_mix 🔹</b>

Mélange et réorganisation
Ce modificateur réarrange l'ordre interne des éléments de votre prompt, comme mélanger les cartes d'un jeu. Avec des valeurs positives, les différentes parties de votre description sont "mélangées", créant des combinaisons inattendues - un personnage pourrait hériter d'attributs destinés au décor. Avec des valeurs négatives, l'effet "démélange" en accentuant les séparations, rendant chaque élément plus distinct. Utile pour créer des hybridations créatives ou au contraire séparer clairement les concepts.
________________________________________

<b>🔸 svd_filter 🔹</b>

Filtrage par complexité (Avancé)
Ce modificateur décompose mathématiquement votre prompt en "composantes de complexité" et les modifie sélectivement. Avec des valeurs positives, il amplifie les détails de niveau moyen, enrichissant les nuances et la sophistication visuelle de votre image. Avec des valeurs négatives, il simplifie le concept en réduisant ces composantes, créant des images plus épurées et minimalistes. Pensez-y comme un filtre qui contrôle la "richesse conceptuelle" de votre génération.
________________________________________

<b>🔸 spherical_rotation 🔹</b>

Rotation conceptuelle (Avancé)
Ce modificateur fait "tourner" votre prompt dans l'espace multidimensionnel des concepts tout en préservant son intensité globale, comme faire pivoter un objet 3D. Avec des valeurs positives élevées, vous obtenez des variations radicales qui gardent le "poids" du prompt original mais explorent des angles complètement différents. Les résultats peuvent être très surprenants car le sujet reste mais son interprétation change dramatiquement. Excellent pour l'exploration créative extrême.
________________________________________

<b>🔸 principal_component 🔹</b>

Modification des axes principaux (Avancé)
Ce modificateur identifie les "axes principaux" de votre prompt (les directions de variation les plus importantes) et les modifie. Avec des valeurs positives, il amplifie ces axes dominants, créant des images qui poussent à l'extrême les caractéristiques principales de votre description. Avec des valeurs négatives, il les atténue, simplifiant l'image en réduisant sa dimensionnalité conceptuelle. C'est comme choisir entre "accentuer ce qui compte le plus" ou "aplatir pour simplifier".
________________________________________

<b>🔸 block_shuffle 🔹</b>

Réorganisation par blocs
Ce modificateur découpe votre prompt en "blocs" conceptuels et les réorganise aléatoirement, tout en préservant la cohérence à l'intérieur de chaque bloc. Avec des valeurs positives croissantes, les blocs deviennent plus petits et le mélange plus chaotique, créant des images surréalistes où les éléments apparaissent dans un ordre inattendu. C'est moins radical que l'embedding_mix car la structure locale est préservée. Parfait pour créer des compositions inhabituelles tout en gardant des éléments reconnaissables.
________________________________________

<b>💡 Conseils généraux d'utilisation</b>

•	Débutants : Commencez avec guided_noise (0.2-0.4) et temperature_scale (0.5-0.7)
•	Variations subtiles : perlin_noise (0.1-0.3), semantic_drift (0.2)
•	Exploration créative : style_shift (0.5-0.8), spherical_rotation (0.6-1.0)
•	Stabilisation : Valeurs négatives sur temperature_scale (-0.3 à -0.5)
•	Effets artistiques : quantize (0.7-1.0), block_shuffle (0.5-0.8)
N'oubliez pas : Changez le seed du node pour obtenir différentes variations avec les mêmes paramètres !

 
<img width="2310" height="900" alt="🍄Great_Conditioning_node" src="https://github.com/user-attachments/assets/1dbc3b63-c14e-49bb-b3ff-c5c2cd0f68c0" />

________________________________________
________________________________________

<b>🍄Great Interactive Gradient Node</b>
![Interactive_Gradient_Node](https://github.com/user-attachments/assets/94572120-eef0-496e-9b32-6506d0a68c2d)


