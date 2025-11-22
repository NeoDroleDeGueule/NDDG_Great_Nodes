import { app } from "/scripts/app.js";

app.registerExtension({
  name: "ImageBlendEditor",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {

    // Vérifie que le node correspond bien à celui qu’on veut modifier
    if (nodeData.name !== "ImageBlendNode") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
      const nodeRef = this;

      // 🎨 Personnalisation des couleurs du node "ImageBlendNode"
      nodeRef.color = "#080808";      // bandeau du titre
      nodeRef.bgcolor = "#353535";    // fond du node
      nodeRef.groupcolor = "#c41c30"; // bande verticale de groupe

      // Exemple : changer le titre affiché (facultatif)
      //nodeRef.title = "🍄Image Blend Node";

      return result;
    };
  },
});


app.registerExtension({
  name: "InteractiveGradientEditor",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {

    if (nodeData.name !== "InteractiveOrganicGradientNode") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
      const nodeRef = this;

      // 🎨 Personnalisation des couleurs du node
      nodeRef.color = "#080808";      // bandeau du titre #d97517 > jaune orangé
      nodeRef.bgcolor = "#353535";    // fond du node
      nodeRef.groupcolor = "#c41c30"; // bande verticale de groupe

      // 🔧 Ajustement propre de la hauteur du node
      const adjustNodeSize = () => {
          try {
              // Trouve la taille actuelle du node
              const baseHeight = nodeRef.size[1];

              // Calcule une nouvelle hauteur minimale (le canvas fait ~520px)
              const minHeight = 630;

              // Si la hauteur actuelle est inférieure, on l'augmente
              if (baseHeight < minHeight) {
                  nodeRef.size[1] = minHeight;
                  nodeRef.setDirtyCanvas(true, true);
              }
          } catch (e) {
              console.warn("⚠️ Impossible d’ajuster la taille du node :", e);
          }
      };

      // Ajustement au chargement + après un court délai
      adjustNodeSize();
      setTimeout(adjustNodeSize, 800);


      // --- Gradient data widget ---
      const gradientWidget = nodeRef.widgets.find(w => w.name === "gradient_data");
      if (!gradientWidget) {
        console.warn("Widget 'gradient_data' non trouvé !");
        return result;
      }

      // --- Container ---
      const container = document.createElement("div");
      container.style.display = "flex";
      container.style.flexDirection = "column";
      container.style.alignItems = "center";
      container.style.gap = "6px";
      container.style.margin = "6px 0";

      // --- Canvas ---
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      canvas.style.width = "100%";
      canvas.style.height = "auto";
      canvas.style.aspectRatio = "1";
      canvas.style.display = "block";
      canvas.style.border = "2px solid " + nodeRef.color;
      canvas.style.cursor = "pointer";
      canvas.style.borderRadius = "6px";
      canvas.style.boxShadow = "0 0 5px rgba(0,0,0,0.4)";
      container.appendChild(canvas);

      // --- Reset button ---
      const resetBtn = document.createElement("button");
      resetBtn.textContent = "Reset Gradient";
      resetBtn.style.padding = "6px 10px";
      resetBtn.style.fontSize = "13px";
      resetBtn.style.border = "2px solid" + nodeRef.color;
      resetBtn.style.borderRadius = "6px";
      resetBtn.style.background = nodeRef.bgcolor;//"#333";
      resetBtn.style.color = "#dddddd";//"white";
      resetBtn.style.cursor = "pointer";
      resetBtn.style.transition = "background 0.2s";
      resetBtn.onmouseenter = () => (resetBtn.style.background = nodeRef.color);
      resetBtn.onmouseleave = () => (resetBtn.style.background = nodeRef.bgcolor);
      container.appendChild(resetBtn);

      // add DOM widget
      const domWidget = nodeRef.addDOMWidget("gradient_editor", "custom", container, {
        onDraw: () => {
          drawFn();
        }
      });

      const ctx = canvas.getContext("2d");

      // --- Data handling ---
      let data = [];
      function defaultGradient() {
        return [
          { x: 0.2, y: 0.5, color: "#ff3300" },
          { x: 0.8, y: 0.5, color: "#00ffe1" }
        ];
      }

      function loadData() {
        try {
          const v = gradientWidget.value ?? "";
          const parsed = v ? JSON.parse(v) : [];
          if (Array.isArray(parsed)) data = parsed;
          else data = defaultGradient();
        } catch (e) {
          data = defaultGradient();
        }
      }
      loadData();

      function saveData() {
        try {
          gradientWidget.value = JSON.stringify(data);
        } catch (e) {
          console.error("Impossible de sauvegarder le gradient:", e);
        }
        try { nodeRef.setDirtyCanvas(true, true); } catch (e) {}
      }

      // Reset handler
      resetBtn.onclick = () => {
        data = defaultGradient();
        saveData();
        drawFn();
      };

      // --- Drawing ---
      function drawFn() {
        if (!Array.isArray(data) || data.length === 0) data = defaultGradient();

        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        // checkerboard
        const tile = 32;
        for (let yy = 0; yy < h; yy += tile) {
          for (let xx = 0; xx < w; xx += tile) {
            ctx.fillStyle = ((xx / tile + yy / tile) % 2 === 0) ? "#cccccc" : "#eeeeee";
            ctx.fillRect(xx, yy, tile, tile);
          }
        }

        // radial blobs
        for (const stop of data) {
          const x = stop.x * w;
          const y = stop.y * h;
          const g = ctx.createRadialGradient(x, y, 0, x, y, Math.max(w, h) / 2);
          g.addColorStop(0, stop.color);
          g.addColorStop(1, "transparent");
          ctx.fillStyle = g;
          ctx.fillRect(0, 0, w, h);
        }

        // handles
        for (const stop of data) {
          const x = stop.x * w;
          const y = stop.y * h;
          ctx.beginPath();
          ctx.arc(x, y, 20, 0, Math.PI * 2);// rayon des pastilles
          ctx.fillStyle = stop.color;
          ctx.fill();
          ctx.strokeStyle = nodeRef.color;
          ctx.lineWidth = 3;
          ctx.stroke();

          // === REMPLACÉ : petit cercle rouge (diamètre 16px => rayon 8px) ===
          const delX = x + 15;
          const delY = y - 15;
          ctx.beginPath();
          ctx.arc(delX, delY, 12, 0, Math.PI * 2); // radius 4px == diameter 8px
          ctx.fillStyle = "red";
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = "#000000";
          ctx.stroke();
        }
      }

      // --- Helpers ---
      function clientToCanvas(eClientX, eClientY) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return { x: (eClientX - rect.left) * scaleX, y: (eClientY - rect.top) * scaleY };
      }

      // --- Interaction ---
      let dragging = false, selected = null, activePointerId = null;

      const onPointerDown = (ev) => {
        if (ev.button && ev.button !== 0) return;
        ev.preventDefault(); ev.stopPropagation();
        loadData();

        const { x, y } = clientToCanvas(ev.clientX, ev.clientY);

        // delete small red circle (use same center as drawing: x+12, y-8)
        for (let i = 0; i < data.length; i++) {
          const s = data[i];
          const px = s.x * canvas.width, py = s.y * canvas.height;
          const cx = px + 15;
          const cy = py - 15;
          // tolerance a bit larger than radius so clicks are easier (use 8px)
          if (Math.hypot(x - cx, y - cy) <= 12) {
            data.splice(i, 1);
            saveData();
            drawFn();
            return;
          }
        }

        // drag handle
        for (const s of data) {
          const px = s.x * canvas.width, py = s.y * canvas.height;
          if (Math.hypot(x - px, y - py) <= 20) {
            dragging = true; selected = s; activePointerId = ev.pointerId;
            try { canvas.setPointerCapture(ev.pointerId); } catch {}
            return;
          }
        }

        // add new
        data.push({ x: x / canvas.width, y: y / canvas.height, color: "#ffffff" });
        saveData(); drawFn();
      };

      const onPointerMove = (ev) => {
        if (!dragging || !selected || ev.pointerId !== activePointerId) return;
        ev.preventDefault(); ev.stopPropagation();
        const { x, y } = clientToCanvas(ev.clientX, ev.clientY);
        selected.x = Math.max(0, Math.min(1, x / canvas.width));
        selected.y = Math.max(0, Math.min(1, y / canvas.height));
        saveData(); drawFn();
      };

      const onPointerUp = (ev) => {
        if (ev.pointerId !== activePointerId) return;
        dragging = false; selected = null;
        try { canvas.releasePointerCapture(ev.pointerId); } catch {}
        activePointerId = null;
      };

      const onDblClick = (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        loadData();
        const { x, y } = clientToCanvas(ev.clientX, ev.clientY);
        for (const s of data) {
          const px = s.x * canvas.width, py = s.y * canvas.height;
          if (Math.hypot(x - px, y - py) <= 10) {
            const input = document.createElement("input");
            input.type = "color"; input.value = s.color;
            input.style.position = "fixed";
            input.style.left = `${ev.clientX}px`;
            input.style.top = `${ev.clientY}px`;
            input.style.zIndex = 99999;
            document.body.appendChild(input);
            input.addEventListener("input", e => { s.color = e.target.value; saveData(); drawFn(); });
            input.addEventListener("change", () => input.remove());
            input.addEventListener("blur", () => input.remove());
            input.focus(); input.click();
            return;
          }
        }
      };

      // attach events
      canvas.addEventListener("pointerdown", onPointerDown, { passive: false });
      canvas.addEventListener("pointermove", onPointerMove, { passive: false });
      canvas.addEventListener("pointerup", onPointerUp, { passive: false });
      canvas.addEventListener("dblclick", onDblClick, { passive: false });

      // initial draw
      saveData(); drawFn();

      return result;
    };
  },
});
