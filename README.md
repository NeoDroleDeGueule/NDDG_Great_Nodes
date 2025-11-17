<b>NDDG Great Nodes</b>

________________________________________
________________________________________

<b>🍄Great Conditioning Modifier</b>


![GreatConditioningModifier](https://github.com/user-attachments/assets/e849c3aa-f770-45c7-9544-6259a66c1aa1)




<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Modifier Guide — Clean Layout</title>
  <style>
    :root{
      --bg:#0f1724; --card:#0b1220; --muted:#9aa4b2; --accent:#6ee7b7; --glass: rgba(255,255,255,0.03);
      --max-width:1100px; --radius:14px;
      color-scheme: dark;
    }
    *{box-sizing:border-box}
    html,body{height:100%;margin:0;font-family:Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background:linear-gradient(180deg,#071021 0%, #071827 60%); color:#e6eef6}
    .wrap{max-width:var(--max-width);margin:40px auto;padding:28px;}
    header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
    header h1{font-size:20px;margin:0;display:flex;gap:12px;align-items:center}
    header p{margin:0;color:var(--muted);font-size:13px}

    .grid{display:grid;grid-template-columns:1fr 340px;gap:20px}
    .main{background:linear-gradient(180deg, rgba(255,255,255,0.02), transparent);padding:22px;border-radius:var(--radius);box-shadow:0 6px 30px rgba(2,6,23,0.6)}
    .sidebar{padding:18px;background:var(--card);border-radius:12px;min-height:200px}

    .intro{margin-bottom:18px}
    .legend{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
    .legend span{padding:8px 10px;border-radius:10px;background:var(--glass);font-size:13px;color:var(--muted)}

    .modifier-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
    .card{background:rgba(255,255,255,0.02);padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,0.02)}
    .card h3{margin:0 0 8px 0;font-size:15px}
    .card p{margin:0;color:var(--muted);line-height:1.45;font-size:14px}
    .card .meta{margin-top:10px;font-size:13px;color:var(--muted)}

    .tips{margin-top:18px;padding:14px;border-radius:12px;background:linear-gradient(180deg, rgba(110,231,183,0.03), rgba(255,255,255,0.01));border:1px solid rgba(110,231,183,0.06)}
    .tips ul{margin:8px 0 0 20px;color:var(--muted)}

    footer{margin-top:18px;color:var(--muted);font-size:13px}

    /* Responsive */
    @media (max-width:900px){.grid{grid-template-columns:1fr}.sidebar{order:2}.main{order:1}}

    /* small helpers */
    .label{display:inline-block;padding:4px 8px;border-radius:8px;background:rgba(255,255,255,0.03);font-size:12px;color:var(--muted)}
    .accent{color:var(--accent)}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1><b>📚 Modifier Guide</b></h1>
      <p>Clean HTML layout of the modifier descriptions — responsive and printable.</p>
    </header>

    <div class="grid">
      <main class="main">
        <section class="intro">
          <div class="legend">
            <span>🔹 &gt; degree of importance for <b>POSITIVE</b> value modifications</span>
            <span>🔸 &gt; degree of importance for <b>NEGATIVE</b> value modifications</span>
            <span>❌ &gt; no use in <b>Positive</b></span>
          </div>
          <p style="color:var(--muted);margin-top:8px">Below are concise, well-structured descriptions for each modifier. Each card explains intent, typical effects for positive and negative values, and suggested use cases.</p>
        </section>

        <section class="modifier-list">

          <article class="card">
            <h3><b>🔸 semantic_drift 🔹</b></h3>
            <p>Progressive semantic drift — gradually blends the original prompt with a noisier version. <br><span class="meta"><b>Positive:</b> gentle drift toward neighboring interpretations. <b>Negative:</b> reinforces prompt, reduces variation. Use for creative variants that keep meaning.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸🔸🔸 token_dropout 🔹🔹</b></h3>
            <p>Selective token removal — randomly ignores tokens to create abstract or surprising outputs. <br><span class="meta"><b>Positive:</b> more abstract images (missing elements). <b>Negative:</b> focuses on few tokens for cleaner results.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸🔸🔸 gradient_amplify 🔹🔹</b></h3>
            <p>Amplifies transitions between prompt elements (conceptual contrast). <br><span class="meta"><b>Positive:</b> stronger contrasts and dramatic images. <b>Negative:</b> smoother, harmonious blends.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸🔸🔸 guided_noise 🔹🔹🔹</b></h3>
            <p>Proportional guided noise — predictable creative noise like film grain. <br><span class="meta"><b>Positive (0.2–0.5):</b> natural variations. <b>Negative:</b> stabilizes and makes images predictable.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸 quantize 🔹🔹🔹🔹</b></h3>
            <p>Quantization and stabilization — reduces instruction precision for a stylized, graphic look. <br><span class="meta"><b>Positive (0.5–1.0):</b> stylized, limited-palette aesthetic. <b>Negative:</b> adds fine dithering and texture.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸🔸🔸 perlin_noise 🔹🔹🔹🔹</b></h3>
            <p>Coherent structured noise — Perlin noise produces natural, flowing variations. <br><span class="meta"><b>Positive:</b> organic, smooth variations. <b>Negative:</b> de-structures patterns to create fragmentation.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸🔸🔸 fourier_filter ❌</b></h3>
            <p>NON-FUNCTIONAL frequency filtering — conceptual low-pass filter (negative values only). <br><span class="meta"><b>Negative:</b> keeps broad shapes and general concepts; smooths fine detail.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸 style_shift 🔹</b></h3>
            <p>Directional style shift — pushes the prompt in a coherent stylistic direction. <br><span class="meta"><b>Positive:</b> explore large stylistic shifts while keeping the subject. <b>Negative:</b> inverse directional shift.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸 temperature_scale 🔹</b></h3>
            <p>Creativity control — like temperature in text models. <br><span class="meta"><b>Positive (0.5–1.0):</b> bolder, less predictable. <b>Negative:</b> conservative and consistent outputs.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸 embedding_mix 🔹</b></h3>
            <p>Mixing and reorganization — rearranges internal order of prompt elements. <br><span class="meta"><b>Positive:</b> unexpected combinations. <b>Negative:</b> separates concepts for clarity.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸 svd_filter 🔹</b></h3>
            <p>Complexity-based filtering (Advanced) — decomposes prompt into components. <br><span class="meta"><b>Positive:</b> amplifies mid-level details. <b>Negative:</b> simplifies for minimalistic results.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸 spherical_rotation 🔹</b></h3>
            <p>Conceptual rotation (Advanced) — rotates prompt in multidimensional concept space. <br><span class="meta"><b>Positive:</b> radical variations that preserve overall weight of the prompt.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸 principal_component 🔹</b></h3>
            <p>Modification of principal axes (Advanced) — identifies and alters main axes of variation. <br><span class="meta"><b>Positive:</b> emphasize dominant features. <b>Negative:</b> attenuate them to simplify.</span></p>
          </article>

          <article class="card">
            <h3><b>🔸 block_shuffle 🔹</b></h3>
            <p>Block-based reorganization — splits prompt into blocks and shuffles them. <br><span class="meta"><b>Positive:</b> smaller blocks and more chaotic shuffles create surreal compositions while preserving local structure.</span></p>
          </article>

        </section>

        <section class="tips">
          <h3 style="margin:0 0 6px 0">💡 General Usage Tips</h3>
          <ul>
            <li>Beginners: Start with <span class="label">guided_noise (0.2–0.4)</span> and <span class="label">temperature_scale (0.5–0.7)</span>.</li>
            <li>Subtle variations: <span class="label">perlin_noise (0.1–0.3)</span>, <span class="label">semantic_drift (0.2)</span>.</li>
            <li>Creative exploration: <span class="label">style_shift (0.5–0.8)</span>, <span class="label">spherical_rotation (0.6–1.0)</span>.</li>
            <li>Stabilization: Negative values on <span class="label">temperature_scale (–0.3 to –0.5)</span>.</li>
            <li>Artistic effects: <span class="label">quantize (0.7–1.0)</span>, <span class="label">block_shuffle (0.5–0.8)</span>.</li>
          </ul>
          <p style="margin-top:8px;color:var(--muted)">Don't forget: Change the seed of the node to get different variations with the same parameters!</p>
        </section>

        <footer>
          <p>Designed for clarity and quick scanning. Use this HTML as a reference panel or paste into a documentation site.</p>
        </footer>

      </main>

      <aside class="sidebar">
        <h4 style="margin:0 0 8px 0">Quick Controls</h4>
        <p style="color:var(--muted);margin-top:0">You can copy the HTML, edit values, or export as a single-file reference.</p>
        <div style="margin-top:12px;display:flex;flex-direction:column;gap:8px">
          <button style="padding:10px;border-radius:10px;border:1px solid rgba(255,255,255,0.04);background:transparent;color:var(--accent);cursor:pointer">Copy HTML</button>
          <button style="padding:10px;border-radius:10px;border:1px solid rgba(255,255,255,0.04);background:transparent;color:var(--muted);cursor:pointer">Print</button>
        </div>

        <div style="margin-top:18px">
          <h5 style="margin:0 0 6px 0">Legend</h5>
          <p style="margin:0;color:var(--muted);font-size:13px">🔹 positive influence — increases variation<br>🔸 negative influence — reduces variation<br>❌ not usable in positive</p>
        </div>
      </aside>
    </div>
  </div>
</body>
</html>


 
<img width="2310" height="900" alt="🍄Great_Conditioning_node" src="https://github.com/user-attachments/assets/1dbc3b63-c14e-49bb-b3ff-c5c2cd0f68c0" />

________________________________________
________________________________________

<b>🍄Great Interactive Gradient Node</b>
![Interactive_Gradient_Node](https://github.com/user-attachments/assets/94572120-eef0-496e-9b32-6506d0a68c2d)


