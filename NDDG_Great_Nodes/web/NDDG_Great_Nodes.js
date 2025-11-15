import { app } from "../../scripts/app.js";

// Configuration des couleurs pour vos nodes
const NODE_COLORS = {
    "KSamplerQwenRandomNoise": {
        nodeColor: "#353535",
        nodeBgColor: "#080808",
        titleTextColor: "#ffffff"
    },
    "QwenToSDLatent": {
        nodeColor: "#353535",
        nodeBgColor: "#080808",
        titleTextColor: "#ffffff"
    },
    "GreatConditioningModifier": {
        nodeColor: "#353535",
        nodeBgColor: "#080808",
        titleTextColor: "#ffffff"
    },
    "ImageBlendNode": {
        nodeColor: "#353535",
        nodeBgColor: "#080808",
        titleTextColor: "#ffffff"
    },
    "InteractiveOrganicGradientNode": {
        nodeColor: "#353535",
        nodeBgColor: "#080808",
        titleTextColor: "#ffffff"
    }
};

app.registerExtension({
    name: "great.custom.node.colors",
    
    async nodeCreated(node) {
        const nodeName = node.constructor.name || node.type;
        const colors = NODE_COLORS[nodeName];
        
        if (!colors) return;
        
        console.log(`[GreatNodes] Applying colors to: ${nodeName}`);
        
        // Méthode moderne pour ComfyUI récent
        node.color = colors.nodeColor;
        node.bgcolor = colors.nodeBgColor;
        
        // Alternative si les propriétés ci-dessus ne marchent pas
        if (node.properties) {
            node.properties.color = colors.nodeColor;
            node.properties.bgcolor = colors.nodeBgColor;
        }
        
        // Force le style via le système de widgets
        const originalOnDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function(ctx) {
            // Appliquer les couleurs au contexte canvas
            const oldFillStyle = ctx.fillStyle;
            const oldStrokeStyle = ctx.strokeStyle;
            
            // Appeler le draw original
            if (originalOnDrawForeground) {
                originalOnDrawForeground.call(this, ctx);
            }
            
            // Restaurer
            ctx.fillStyle = oldFillStyle;
            ctx.strokeStyle = oldStrokeStyle;
        };
        
        // Forcer le redraw
        node.setDirtyCanvas?.(true, true);
    }
});


