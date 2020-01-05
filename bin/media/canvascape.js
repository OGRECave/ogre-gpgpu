/**
 * Copyright (c) 2009, Benjamin Joffe (http://www.benjoffe.com/code/demos/canvascape)
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AUTHOR AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

var map;
var canvas;
var overlay;
//variables initiated at the bottom of the code...

var pi=Math.PI;

var total=0;

var total=0;

var samples=200;


var arena=[];
arena[0]=[1,1,1,1,1,1,1,1,1,1]
arena[1]=[1,0,0,0,0,0,0,0,0,1]
arena[2]=[1,0,0,1,0,1,1,1,0,1]
arena[3]=[1,0,1,0,0,0,0,1,0,1]
arena[4]=[1,0,0,0,0,1,0,1,0,1]
arena[5]=[1,0,1,1,0,0,0,0,0,1]
arena[6]=[1,0,0,1,0,1,1,1,0,1]
arena[7]=[1,1,0,1,0,0,0,1,0,1]
arena[8]=[1,0,0,1,0,1,0,0,0,1]
arena[9]=[1,1,1,1,1,1,1,1,1,1]


var playerPos=[4,4]; // x,y (from top left)
var playerDir=0.4; // theta, facing right=0=2pi
var playerPosZ=1;
var key=[0,0,0,0,0]; // left, right, up, down

var playerVelY=0;


var face=[];

function wallDistance(theta){

	var data=[];
	face=[];

	var x = playerPos[0], y = playerPos[1];
	var deltaX, deltaY;
	var distX, distY;
	var stepX, stepY;
	var mapX, mapY
	
	var atX=Math.floor(x), atY=Math.floor(y);

	var thisRow=-1;
	var thisSide=-1;

	var lastHeight=0;

	for (var i=0; i<samples; i++) {
		theta+=pi/(3*samples)+2*pi;
		theta%=2*pi;

		mapX = atX, mapY = atY;

		deltaX=1/Math.cos(theta);
		deltaY=1/Math.sin(theta);

		if (deltaX>0) {
			stepX = 1;
			distX = (mapX + 1 - x) * deltaX;
		}
		else {
			stepX = -1;
			distX = (x - mapX) * (deltaX*=-1);		
		}
		if (deltaY>0) {
			stepY = 1;
			distY = (mapY + 1 - y) * deltaY;
		}
		else {
			stepY = -1;
			distY = (y - mapY) * (deltaY*=-1);
		}
		

		for (var j=0; j<20; j++) {
			if (distX < distY) {
				mapX += stepX;
				if (arena[mapX][mapY]) {
					if (thisRow!=mapX || thisSide!=0) {
						if (i>0) {
							data.push(i);
							data.push(lastHeight);
						}
						data.push(i);
						data.push(distX);
						thisSide=0;
						thisRow=mapX;
						face.push(1+stepX);
					}
					lastHeight=distX;
					break;
				}
				distX += deltaX;
			}
			else {
				mapY += stepY;
				if (arena[mapX][mapY]) {
					if (thisRow!=mapY || thisSide!=1) {
						if (i>0) {
							data.push(i);
							data.push(lastHeight);
						}	
						data.push(i);
						data.push(distY);
						thisSide=1;
						thisRow=mapY;
						face.push(2+stepY)
					}
					lastHeight=distY;
					break;
				}
				distY += deltaY;
			}
		}
	}
	data.push(i);
	data.push(lastHeight);
	
	return data;
}

function clearCanvas() {
	ctx.save();
	ctx.fillStyle = "white";
	ctx.fillRect(0, 0, 400, 300);
	
	var grad = ctx.createLinearGradient(0, 150, 0, 300);
	grad.addColorStop(0.5, '#263626');
	grad.addColorStop(0.75, '#626a62');	
	grad.addColorStop(1, '#989898');	
	ctx.fillStyle = grad;
	ctx.fillRect(0, 150, 400, 150);		
	
	ctx.restore();
	grad.destroy();
}

function drawCanvas(){

	clearCanvas();

	var theta = playerDir-pi/6;

	var wall=wallDistance(theta);


	for (var i=0; i<wall.length; i+=4) {

		var tl,tr,bl,br;
		
		var theta1,theta2,fix1,fix2;	
	
		theta1=playerDir-pi/6 + pi*wall[i]/(3*samples);
		theta2=playerDir-pi/6 + pi*wall[i+2]/(3*samples);
		
		fix1 = Math.cos(theta1-playerDir);
		fix2 = Math.cos(theta2-playerDir);

		var h=2-playerPosZ;

		var wallH1=100/(wall[i+1]*fix1);
		var wallH2=100/(wall[i+3]*fix2);
		
		tl=[wall[i]*2, 150-wallH1*h];
		tr=[wall[i+2]*2, 150.0-parseFloat(wallH2)*parseFloat(h)]
		br=[wall[i+2]*2, tr[1]+wallH2*2];
		bl=[wall[i]*2, tl[1]+wallH1*2]

		var shade1=Math.floor(wallH1*2+20); if (shade1>255) shade1=255;
		var shade2=Math.floor(wallH2*2+20); if (shade2>255) shade2=255;

		
		var linGrad = ctx.createLinearGradient(tl[0],0,tr[0],0);
		linGrad.addColorStop(0, 'rgba('+(face[i/4]%2==0 ? shade1 : 0)+','+(face[i/4]==1 ? shade1 : 0)+','+(face[i/4]==2 ? 0 : shade1)+',1.0)');
		linGrad.addColorStop(1, 'rgba('+(face[i/4]%2==0 ? shade2 : 0)+','+(face[i/4]==1 ? shade2 : 0)+','+(face[i/4]==2 ? 0 : shade2)+',1.0)');

		ctx.beginPath();
		ctx.moveTo(tl[0], tl[1]);
		ctx.lineTo(tr[0], tr[1]);
		ctx.lineTo(br[0], br[1]);
		ctx.lineTo(bl[0], bl[1]);
		ctx.save();
		ctx.fillStyle = linGrad;
		ctx.fill();
		ctx.restore();
		linGrad.destroy();
	}	
}

function nearWall(x,y){
	var xx,yy;
	if (isNaN(x)) x=playerPos[0];
	if (isNaN(y)) y=playerPos[1];
	for (var i=-0.1; i<=0.1; i+=0.2) {
		xx=Math.floor(x+i)
		for (var j=-0.1; j<=0.1; j+=0.2) {
			yy=Math.floor(y+j);
			if (arena[xx][yy]) return true;
		}
	}
	return false;
}

clearCanvas();
drawCanvas();


function update(){

	total++;

	var change=false;
/*
	if (jumpCycle) {
		jumpCycle--;
		change=true;
		playerPosZ = 1 + jumpCycle*(20-jumpCycle)/110;
	}
	else if (key[4]) jumpCycle=20;
*/
	if (key[0]) {
		if (!key[1]) {
			playerDir-=0.07; //left
			change=true;
		}
	}
	else if (key[1]) {
		playerDir+=0.07; //right
		change=true;
	}
/*
	if (change) {
		playerDir+=2*pi;
		playerDir%=2*pi;
		document.getElementById("sky").style.backgroundPosition=Math.floor(1-playerDir/(2*pi)*2400)+"px 0";
	}
*/
	if (key[2] && !key[3]) {
		if (playerVelY<0.1) playerVelY += 0.02;
	}
	else if (key[3] && !key[2]) {
		if (playerVelY>-0.1) playerVelY -= 0.02;
	}
	else {
		if (playerVelY<-0.02) playerVelY += 0.015;
		else if (playerVelY>0.02) playerVelY -= 0.015;
		else playerVelY=0;
	}
	
	
	if (playerVelY!=0) {

		var oldX=playerPos[0];;
		var oldY=playerPos[1];		
		var newX=oldX+Math.cos(playerDir)*playerVelY;
		var newY=oldY+Math.sin(playerDir)*playerVelY;

		if (!nearWall(newX, oldY)) {
			playerPos[0]=newX;
			oldX=newX;
			change=true;
		}
		if (!nearWall(oldX, newY)) {
			playerPos[1]=newY;
			change=true;
		}

	}
	if (change) drawCanvas();
}
/*

function changeKey(which, to){
	switch (which){
		case 65:case 37: key[0]=to; break; // left
		case 87: case 38: key[2]=to; break; // up
		case 68: case 39: key[1]=to; break; // right
		case 83: case 40: key[3]=to; break;// down
		case 32: key[4]=to; break; // space bar;
		case 17: key[5]=to; break; // ctrl
		case 66: if (to) { shoot() } break; // b
	}
}
document.onkeydown=function(e){changeKey((e||window.event).keyCode, 1);}
document.onkeyup=function(e){changeKey((e||window.event).keyCode, 0);}


function initUnderMap(){
	var underMap=document.getElementById("underMap").getContext("2d");
	underMap.fillStyle="#FFF";
	underMap.fillRect(0,0, 200, 200);
	underMap.fillStyle="#444";
	for (var i=0; i<arena.length; i++) {
		for (var j=0; j<arena[i].length; j++) {
			if (arena[i][j]) underMap.fillRect(i*8, j*8, 8, 8);
		}	
	}
}

window.onerror=function(){
	alert('An error has occured, the most likely reason is because you are using an incompatible browser.\nYou must be using one of the following browsers or a newer version:\n\n- Internet Explorer 6\n- Firefox 1.5\n- Safari 1.3\n- Opera 9');
	window.onerror=function(){};
	return true;
}

window.onload=function(){
	map=document.getElementById("map").getContext("2d");
	canvas=document.getElementById("canvas").getContext("2d");
	overlay=document.getElementById("overlay");
	document.getElementById("sky").style.backgroundPosition=Math.floor(-playerDir/(2*pi)*2400)+"px 0";
	drawCanvas();
	initUnderMap();
	setInterval(update, 35);
}
*/