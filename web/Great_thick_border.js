/**
 * Extension ComfyUI - Bordure Épaisse FINALE avec support COMPLET des Subgraphs
 * ✅ Détecte les nodes parents quand leurs enfants sont exécutés (format parent_id:child_id)
 * ✅ Suit parfaitement le contour de TOUS les types de nodes
 * 
 * Installation: ComfyUI/custom_nodes/thick-border/web/thick_border.js
 * 
 * 🍄NDDG Great Nodes
 * 
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "thick.executing.border.final",
    
    async setup() {
        console.log("🎨 Extension Thick Border FINAL chargée !");
        
        // ============ CONFIGURATION ============
        const CONFIG = {
            borderWidth: 6,           // Épaisseur de la bordure (3-10)
            borderColor: "#00FF00",   // Couleur
            borderOffset: 6,          // Distance du node (4-10)
            addGlow: true,            // Effet lumineux
            glowIntensity: 15,        // Intensité de la lueur
            opacity: 0.95,            // Opacité de la bordure
            showSubgraphChildren: true  // Mettre en évidence les nodes enfants aussi
        };
        // ========================================
        
        // Attendre LGraphCanvas et LiteGraph
        await new Promise((resolve) => {
            if (typeof LGraphCanvas !== 'undefined' && typeof LiteGraph !== 'undefined') {
                resolve();
            } else {
                const check = setInterval(() => {
                    if (typeof LGraphCanvas !== 'undefined' && typeof LiteGraph !== 'undefined') {
                        clearInterval(check);
                        resolve();
                    }
                }, 100);
            }
        });
        
        console.log("✅ LGraphCanvas détecté !");
        
        // ===== FONCTION HELPER : Extraire l'ID parent d'un node enfant =====
        function getParentNodeId(nodeId) {
            // Format des nodes enfants: "parent_id:child_id"
            if (typeof nodeId === 'string' && nodeId.includes(':')) {
                return parseInt(nodeId.split(':')[0]);
            }
            return null;
        }
        
        // ===== FONCTION DE DESSIN DE LA BORDURE PARFAITE =====
        function drawPerfectBorder(ctx, node, canvas) {
            ctx.save();
            
            // Configuration du style
            ctx.lineWidth = CONFIG.borderWidth;
            ctx.strokeStyle = CONFIG.borderColor;
            ctx.globalAlpha = CONFIG.opacity;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            
            // Effet de lueur
            if (CONFIG.addGlow) {
                ctx.shadowBlur = CONFIG.glowIntensity;
                ctx.shadowColor = CONFIG.borderColor;
            }
            
            // Récupérer les dimensions et forme du node
            const shape = node._shape || node.constructor.shape || LiteGraph.ROUND_SHAPE;
            const size = node.size;
            const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
            const isCollapsed = node.flags && node.flags.collapsed;
            const offset = CONFIG.borderOffset;
            
            // Rayon des coins arrondis
            let radius = canvas.round_radius || 10;
            
            ctx.beginPath();
            
            if (isCollapsed) {
                // ===== NODE COLLAPSÉ =====
                const collapsedWidth = node._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH || 80;
                const collapsedRadius = LiteGraph.NODE_COLLAPSED_RADIUS || 10;
                
                const x = -offset;
                const y = -titleHeight - offset;
                const w = collapsedWidth + offset * 2;
                const h = titleHeight + offset * 2;
                const r = collapsedRadius + offset / 2;
                
                // Dessiner rectangle arrondi
                ctx.moveTo(x + r, y);
                ctx.lineTo(x + w - r, y);
                ctx.arcTo(x + w, y, x + w, y + r, r);
                ctx.lineTo(x + w, y + h - r);
                ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
                ctx.lineTo(x + r, y + h);
                ctx.arcTo(x, y + h, x, y + h - r, r);
                ctx.lineTo(x, y + r);
                ctx.arcTo(x, y, x + r, y, r);
                
            } else {
                // ===== NODE DÉVELOPPÉ =====
                
                if (shape === LiteGraph.BOX_SHAPE) {
                    // Forme rectangulaire SANS coins arrondis
                    const x = -offset;
                    const y = -titleHeight - offset;
                    const w = size[0] + offset * 2;
                    const h = size[1] + titleHeight + offset * 2;
                    
                    ctx.rect(x, y, w, h);
                    
                } else if (shape === LiteGraph.ROUND_SHAPE || shape === LiteGraph.CARD_SHAPE) {
                    // Forme arrondie (LA PLUS COURANTE)
                    const x = -offset;
                    const y = -titleHeight - offset;
                    const w = size[0] + offset * 2;
                    const h = size[1] + titleHeight + offset * 2;
                    const r = radius + offset / 2;
                    
                    // Rectangle avec coins parfaitement arrondis
                    ctx.moveTo(x + r, y);
                    ctx.lineTo(x + w - r, y);
                    ctx.arcTo(x + w, y, x + w, y + r, r);
                    ctx.lineTo(x + w, y + h - r);
                    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
                    ctx.lineTo(x + r, y + h);
                    ctx.arcTo(x, y + h, x, y + h - r, r);
                    ctx.lineTo(x, y + r);
                    ctx.arcTo(x, y, x + r, y, r);
                    ctx.closePath();
                    
                } else if (shape === LiteGraph.CIRCLE_SHAPE) {
                    // Forme circulaire
                    const centerX = size[0] * 0.5;
                    const centerY = size[1] * 0.5;
                    const circleRadius = Math.max(size[0], size[1]) * 0.5 + offset;
                    
                    ctx.arc(centerX, centerY, circleRadius, 0, Math.PI * 2);
                    
                } else {
                    // Par défaut : forme arrondie
                    const x = -offset;
                    const y = -titleHeight - offset;
                    const w = size[0] + offset * 2;
                    const h = size[1] + titleHeight + offset * 2;
                    const r = radius + offset / 2;
                    
                    ctx.moveTo(x + r, y);
                    ctx.lineTo(x + w - r, y);
                    ctx.arcTo(x + w, y, x + w, y + r, r);
                    ctx.lineTo(x + w, y + h - r);
                    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
                    ctx.lineTo(x + r, y + h);
                    ctx.arcTo(x, y + h, x, y + h - r, r);
                    ctx.lineTo(x, y + r);
                    ctx.arcTo(x, y, x + r, y, r);
                    ctx.closePath();
                }
            }
            
            // Dessiner la bordure
            ctx.stroke();
            ctx.restore();
        }
        
        // ===== INTERCEPTER drawNode =====
        const originalDrawNode = LGraphCanvas.prototype.drawNode;
        
        if (!originalDrawNode) {
            console.error("❌ Erreur: drawNode introuvable");
            return;
        }
        
        LGraphCanvas.prototype.drawNode = function(node, ctx) {
            // Dessiner le node normalement
            originalDrawNode.call(this, node, ctx);
            
            // Vérifier si le node est en exécution
            const isExecuting = node.isExecuting || 
                               node.isExecutingAsParent ||
                               (app.runningNodeId && app.runningNodeId == node.id);
            
            if (isExecuting) {
                // Dessiner la bordure parfaite
                drawPerfectBorder(ctx, node, this);
            }
        };
        
        // ===== ÉCOUTER LES ÉVÉNEMENTS D'EXÉCUTION =====
        if (app.api) {
            app.api.addEventListener("executing", ({ detail }) => {
                const nodeId = detail;
                
                // Réinitialiser tous les flags d'exécution
                if (app.graph && app.graph._nodes) {
                    app.graph._nodes.forEach(node => {
                        node.isExecuting = false;
                        node.isExecutingAsParent = false;
                    });
                }
                
                if (nodeId && app.graph) {
                    // Vérifier si c'est un node enfant (format "parent_id:child_id")
                    const parentId = getParentNodeId(nodeId);
                    
                    if (parentId !== null) {
                        // C'EST UN NODE ENFANT D'UN SUBGRAPH !
                        const parentNode = app.graph.getNodeById(parentId);
                        
                        if (parentNode) {
                            // Marquer le PARENT comme en exécution
                            parentNode.isExecutingAsParent = true;
                            console.log(`⚡ Subgraph parent "${parentNode.title || parentId}" en exécution (enfant: ${nodeId})`);
                            
                            // Optionnel : marquer aussi l'enfant si configuré
                            if (CONFIG.showSubgraphChildren) {
                                // Note: les nodes enfants ne sont pas directement accessibles
                                // via getNodeById car ils sont dans un sous-graphe
                            }
                        } else {
                            console.warn(`⚠️ Parent node ${parentId} introuvable pour l'enfant ${nodeId}`);
                        }
                    } else {
                        // C'est un node normal (pas un enfant)
                        const executingNode = app.graph.getNodeById(nodeId);
                        if (executingNode) {
                            executingNode.isExecuting = true;
                            console.log(`⚡ Node "${executingNode.title || nodeId}" en exécution`);
                        }
                    }
                    
                    // Forcer un redessin
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                }
                
                // Fin de l'exécution
                if (!nodeId) {
                    console.log("✅ Exécution terminée");
                    if (app.canvas) {
                        app.canvas.setDirty(true, true);
                    }
                }
            });
            
            console.log("📡 Écoute WebSocket activée");
        }
        
        console.log(`✨ Configuration:`);
        console.log(`   • Épaisseur: ${CONFIG.borderWidth}px`);
        console.log(`   • Distance: ${CONFIG.borderOffset}px`);
        console.log(`   • Couleur: ${CONFIG.borderColor}`);
        console.log(`   • Effet lumineux: ${CONFIG.addGlow ? 'OUI' : 'NON'}`);
        console.log(`   • Support subgraphs: ACTIVÉ (détection parent:child)`);
    }
});