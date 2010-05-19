// create new image object to use as pattern
var img = ctx.loadImage('wallpaper.png');

// create pattern
var ptrn = ctx.createPattern(img, 'repeat');
ctx.fillStyle = ptrn;
ctx.fillRect(0,0,150,150);