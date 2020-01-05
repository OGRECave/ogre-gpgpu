var i = 0;
var sin = Math.sin(Math.PI/6);
var cos = Math.cos(Math.PI/6);
ctx.translate(200, 200);
var c = 0;
for (i; i <= 12; i++) {
  c = Math.floor(255 / 12 * i);
  ctx.fillStyle = "rgb(" + c + "," + c + "," + c + ")";
  ctx.fillRect(0, 0, 100, 10);
  ctx.transform(cos, sin, -sin, cos, 0, 0);
}

ctx.setTransform(-1, 0, 0, 1, 200, 200);
ctx.fillStyle = "rgba(255, 128, 255, 0.5)";
ctx.fillRect(0, 50, 100, 100);