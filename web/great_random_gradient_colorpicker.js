import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Créer un widget color picker personnalisé
function colorWidget(node, inputName, inputData, app) {
    const widget = {
        type: "JOVI_COLOR",
        name: inputName,
        value: inputData[1]?.default || widget.value || "#c7c7c7",
        options: {},
        
        draw: function(ctx, node, width, y) {
            const margin = 15;
            const labelWidth = ctx.measureText(inputName).width + 5;
            const colorX = margin + labelWidth;
            const colorWidth = width - colorX - margin;

            if (colorWidth < 10) {
                // Fallback : petite pastille à droite
                const colorBoxSize = 20;
                const x = width - colorBoxSize - margin;
                ctx.fillStyle = "#AAA";
                ctx.font = "12px Arial";
                ctx.fillText(inputName, margin, y + 14);
                ctx.fillStyle = this.value || "#c7c7c7";
                ctx.fillRect(x, y + 2, colorBoxSize, colorBoxSize);
                ctx.strokeStyle = "#ffffff";
                ctx.strokeRect(x, y + 2, colorBoxSize, colorBoxSize);
                this._clickArea = { x, width: colorBoxSize, y: y + 2, height: 20 };
                return;
            }

            ctx.fillStyle = "#AAA";
            ctx.font = "12px Arial";
            ctx.fillText(inputName, margin, y + 14);
            
            ctx.fillStyle = this.value || "#c7c7c7";
            ctx.fillRect(colorX, y + 2, colorWidth, 20);
            
            ctx.strokeStyle = "#ffffff";
            ctx.strokeRect(colorX, y + 2, colorWidth, 20);

            this._clickArea = {
                x: colorX,
                width: colorWidth,
                y: y + 2,
                height: 20
            };
        },
        
        mouse: function(event, pos, node) {
            const area = this._clickArea || {};
            if (event.type === "pointerdown" && 
                pos[0] >= area.x && 
                pos[0] <= area.x + area.width &&
                pos[1] >= area.y && 
                pos[1] <= area.y + area.height) {
                
                if (!this.inputEl) {
                    this.inputEl = document.createElement("input");
                    this.inputEl.type = "color";
                    this.inputEl.style.position = "absolute";
                    this.inputEl.style.opacity = "0";
                    this.inputEl.style.pointerEvents = "none";
                    this.inputEl.style.zIndex = "-1";
                    document.body.appendChild(this.inputEl);
                    
                    this.inputEl.addEventListener("change", (e) => {
                        this.value = e.target.value;
                        node.graph.setDirtyCanvas(true, false);
                    });
                }
                
                if (!this.inputEl.parentNode) {
                    document.body.appendChild(this.inputEl);
                }
                
                this.inputEl.value = this.value;
                this.inputEl.focus();
                this.inputEl.click();
                
                return true;
            }
            return false;
        },
        
        computeSize: function(width) {
            return [width, 25];
        }
    };
    
    // ⚠️ NE PAS ajouter ici — on gère l'ajout manuellement dans beforeRegisterNodeDef
    // node.addCustomWidget(widget);
    
    return widget;
}

ComfyWidgets.JOVI_COLOR = colorWidget;

app.registerExtension({
    name: "great.random.gradient.colorpicker",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "GreatRandomOrganicGradientNode") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            const colorWidgets = ["color1", "color2", "color3", "color4", 
                                 "color5", "color6", "color7", "color8", 
                                 "background_color"];
            
            const newWidgets = [];
            for (let i = 0; i < this.widgets.length; i++) {
                const widget = this.widgets[i];
                if (colorWidgets.includes(widget.name)) {
                    const newWidget = colorWidget(this, widget.name, [null, {default: widget.value}], app);
                    newWidget.value = widget.value;
                    newWidgets.push(newWidget);
                } else {
                    newWidgets.push(widget);
                }
            }
            
            this.widgets = newWidgets;
            this.setSize(this.computeSize());
            
            return result;
        };
    }
});