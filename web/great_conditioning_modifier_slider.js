// d'après > ComfyUI.mxToolkit.Slider v.0.9.92 - Max Smirnov 2025
// Great Conditioning Modifier - Custom Slider v1.0
import { app } from "../../scripts/app.js";

class GreatSlider {
    constructor(node, widgetName, config = {}) {
        this.node = node;
        this.widgetName = widgetName;
        this.config = {
            min: config.min || -10.0,
            max: config.max || 10.0,
            step: config.step || 0.01,
            decimals: config.decimals || 2,
            default: config.default || 0.0
        };

        this.value = config.default;
        this.normalizedPos = this.valueToPosition(this.value);

        this.isDragging = false;
        this.startY = 0;
    }

    valueToPosition(value) {
        return (value - this.config.min) / (this.config.max - this.config.min);
    }

    positionToValue(pos) {
        const value = this.config.min + (this.config.max - this.config.min) * pos;
        return Math.round(value / this.config.step) * this.config.step;
    }

    setValue(value) {
        this.value = Math.max(this.config.min, Math.min(this.config.max, value));
        this.normalizedPos = this.valueToPosition(this.value);
    }

    draw(ctx, y, width) {
        const margin = 10;
        const sliderWidth = width - 75;
        const sliderHeight = 10;
        const centerY = y + 15;

        ctx.fillStyle = "rgba(30, 30, 30, 0.8)";
        ctx.beginPath();
        ctx.roundRect(margin, centerY - sliderHeight/2, sliderWidth, sliderHeight, 5);
        ctx.fill();

        const fillWidth = sliderWidth * this.normalizedPos;
        if (fillWidth >= 0) {
            const gradient = ctx.createLinearGradient(margin, 0, margin + sliderWidth, 0);

            if (this.value < 0) {
                gradient.addColorStop(0, "#f0db1f");
                gradient.addColorStop(0.5, "#944212");
                gradient.addColorStop(1, "#000");
            } else if (this.value > 0) {
                gradient.addColorStop(0, "#000");
                gradient.addColorStop(0.5, "#114e80");
                gradient.addColorStop(1, "#21eff3");
            } else {
                gradient.addColorStop(0, "#a2a2a2");
                gradient.addColorStop(1, "#a2a2a2");
            }

            ctx.fillStyle = gradient;
            ctx.beginPath();

            if (this.value < 0) {
                ctx.roundRect(
                    margin + sliderWidth,
                    centerY - sliderHeight/2,
                    (-sliderWidth /2) + ((-sliderWidth / 2) + ((sliderWidth) * this.normalizedPos)),
                    sliderHeight, 5
                );
            } else {
                ctx.roundRect(
                    margin, centerY - sliderHeight/2,
                    fillWidth, sliderHeight, 5
                );
            }

            ctx.fill();
        }

        const zeroPos = margin + sliderWidth * this.valueToPosition(0);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(zeroPos, centerY - sliderHeight);
        ctx.lineTo(zeroPos, centerY + sliderHeight);
        ctx.stroke();

        const cursorX = margin + sliderWidth * this.normalizedPos;
        ctx.fillStyle = "#a2a2a2";
        ctx.beginPath();
        ctx.arc(cursorX, centerY, 8, 0, 2 * Math.PI);
        ctx.fill();

        ctx.strokeStyle = this.node.bgcolor || "#333333";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cursorX, centerY, 6, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = "18px monospace";
        ctx.textAlign = "right";
        ctx.fillText(this.value.toFixed(this.config.decimals), width - 6, centerY + 4);
    }

    handleMouseDown(e, localX, localY, width) {
        const margin = 10;
        const sliderWidth = width - 75;

        if (localY >= 5 && localY <= 25) {
            if (localX >= margin && localX <= margin + sliderWidth) {
                this.isDragging = true;
                this.updateFromMouse(localX, width);
                return true;
            }

            if (localX >= width - 65 && localX <= width - 5) {
                return "prompt";
            }
        }
        return false;
    }

    handleMouseMove(e, localX, localY, width) {
        if (this.isDragging) {
            this.updateFromMouse(localX, width);
            return true;
        }
        return false;
    }

    handleMouseUp(e) {
        if (this.isDragging) {
            this.isDragging = false;
            return true;
        }
        return false;
    }

    updateFromMouse(localX, width) {
        const margin = 10;
        const sliderWidth = width - 75;

        let pos = (localX - margin) / sliderWidth;
        pos = Math.max(0, Math.min(1, pos));

        const newValue = this.positionToValue(pos);

        if (newValue !== this.value) {
            this.setValue(newValue);

            const strengthWidget = this.node.widgets?.find(w => w.name === this.widgetName);
            if (strengthWidget) strengthWidget.value = this.value;

            return true;
        }
        return false;
    }
}

app.registerExtension({
    name: "great.conditioning.modifier.slider",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "GreatConditioningModifier") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {

            const result = onNodeCreated?.apply(this, arguments);

            const strengthWidget = this.widgets?.find(w => w.name === "modification_strength");

            const savedValue = strengthWidget.value;

            if (strengthWidget) {
                strengthWidget.hidden = true;
                strengthWidget.type = "hidden";
            }

            this.strengthSlider = new GreatSlider(this, "modification_strength", {
                min: -10.0,
                max: 10.0,
                step: 0.01,
                decimals: 2,
                default: savedValue
            });

            this.strengthSlider.setValue(savedValue);

            this.sliderHeight = 40;

            const originalComputeSize = this.computeSize;
            this.computeSize = function(out) {
                const size = originalComputeSize ? originalComputeSize.call(this, out) : [this.size[0], this.size[1]];
                size[1] += this.sliderHeight + 10 || 0;
                return size;
            };

            this.size = this.computeSize();

            // ✅✅✅ PATCH CRUCIAL : récupère la vraie valeur chargée par le workflow
            const originalOnConfigure = this.onConfigure;
            this.onConfigure = function(info) {
                const r = originalOnConfigure?.apply(this, arguments);

                const w = this.widgets?.find(w => w.name === "modification_strength");
                if (w && this.strengthSlider) {
                    this.strengthSlider.setValue(w.value);
                }

                return r;
            };
            // ✅✅✅ FIN DU PATCH

            return result;
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            const result = onDrawForeground?.apply(this, arguments);

            if (this.flags.collapsed) return result;

            if (this.strengthSlider) {
                const sliderY = this.size[1] - this.sliderHeight + 5;

                ctx.save();
                this.strengthSlider.draw(ctx, sliderY, this.size[0]);
                ctx.restore();

                const strengthWidget = this.widgets?.find(w => w.name === "modification_strength");
                if (strengthWidget && strengthWidget.value !== this.strengthSlider.value) {
                    strengthWidget.value = this.strengthSlider.value;
                    if (this.onPropertyChanged) {
                        this.onPropertyChanged("modification_strength", this.strengthSlider.value);
                    }
                }
            }

            return result;
        };

        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function(e, localPos, canvas) {
            if (this.strengthSlider) {
                const sliderY = this.size[1] - this.sliderHeight + 5;
                const localX = localPos[0];
                const localY = localPos[1] - sliderY;

                const result = this.strengthSlider.handleMouseDown(e, localX, localY, this.size[0]);

                if (result === "prompt") {
                    canvas.prompt(
                        "Modification Strength",
                        this.strengthSlider.value,
                        (v) => {
                            const num = parseFloat(v);
                            if (!isNaN(num)) {
                                this.strengthSlider.setValue(num);
                                this.setDirtyCanvas(true, true);
                            }
                        },
                        e
                    );
                    return true;
                }

                if (result) {
                    this.setDirtyCanvas(true, true);
                    return true;
                }
            }

            return onMouseDown?.apply(this, arguments);
        };

        const onMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function(e, localPos, canvas) {
            if (this.strengthSlider && this.strengthSlider.isDragging) {
                const sliderY = this.size[1] - this.sliderHeight + 5;
                const localX = localPos[0];
                const localY = localPos[1] - sliderY;

                if (this.strengthSlider.handleMouseMove(e, localX, localY, this.size[0])) {
                    this.setDirtyCanvas(true, true);
                    return true;
                }
            }

            return onMouseMove?.apply(this, arguments);
        };

        const onMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function(e, localPos, canvas) {
            if (this.strengthSlider) {
                if (this.strengthSlider.handleMouseUp(e)) {
                    this.setDirtyCanvas(true, true);
                    return true;
                }
            }

            return onMouseUp?.apply(this, arguments);
        };
    }
});