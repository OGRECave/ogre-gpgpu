var img = ctx.loadImage('rhino.jpg');

for (i=0;i<4;i++){  
	for (j=0;j<3;j++){  
		ctx.drawImage(img,j*50,i*38,50,38);  
	}  
}  