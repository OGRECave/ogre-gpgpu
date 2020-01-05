/*
	Copyright (c) 2010 ASTRE Henri (http://www.visual-experiments.com)

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
*/

#pragma once

#include <string>
#include <v8.h>

#include "CanvasContext.h"
#include "CanvasLogger.h"
#include "CanvasV8Context.h"

//Helper
std::string toString(v8::Local<v8::Value> value);
Ogre::Canvas::Logger* getCanvasLoggerPointer(const v8::Arguments& args);
Ogre::Canvas::Gradient* getGradientPointer(const v8::Arguments& args);
Ogre::Canvas::Context* getCanvasContextPointer(const v8::Arguments& args);
Ogre::Canvas::Context* getCanvasContextPointer(const v8::AccessorInfo& info);
Ogre::Canvas::V8Context* getV8CanvasContext(const v8::Arguments& args);

Ogre::ColourValue getColor(v8::Local<v8::Value> value);
Ogre::Canvas::Gradient* getGradient(v8::Local<v8::Value> value);
Ogre::Canvas::Pattern* getPattern(v8::Local<v8::Value> value);
Ogre::Image* getImage(v8::Local<v8::Value> value);

//Console
v8::Handle<v8::Value> log(const v8::Arguments& args);

//Image
v8::Handle<v8::Value> loadImage(const v8::Arguments& args);

//Canvas Gradient
v8::Handle<v8::Value> addColorStop(const v8::Arguments& args);
v8::Handle<v8::Value> gradient_destroy(const v8::Arguments& args);

//2D Context
v8::Handle<v8::Value> save(const v8::Arguments& args);
v8::Handle<v8::Value> restore(const v8::Arguments& args);

//Transformation
v8::Handle<v8::Value> scale(const v8::Arguments& args);
v8::Handle<v8::Value> rotate(const v8::Arguments& args);
v8::Handle<v8::Value> translate(const v8::Arguments& args);
v8::Handle<v8::Value> transform(const v8::Arguments& args);
v8::Handle<v8::Value> setTransform(const v8::Arguments& args);

//Image drawing
v8::Handle<v8::Value> drawImage(const v8::Arguments& args);

//Compositing
v8::Handle<v8::Value> getterGlobalAlpha(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterGlobalAlpha(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterGlobalCompositeOperation(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterGlobalCompositeOperation(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

//Line styles
v8::Handle<v8::Value> getterLineWidth(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterLineWidth(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterLineCap(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterLineCap(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterLineJoin(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterLineJoin(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterMiterLimit(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterMiterLimit(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterLineDash(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterLineDash(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

//Colors, styles and shadows
v8::Handle<v8::Value> getterStrokeStyle(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterStrokeStyle(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterFillStyle(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterFillStyle(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterShadowOffsetX(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterShadowOffsetX(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterShadowOffsetY(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterShadowOffsetY(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterShadowBlur(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterShadowBlur(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterShadowColor(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterShadowColor(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> createLinearGradient(const v8::Arguments& args);
v8::Handle<v8::Value> createRadialGradient(const v8::Arguments& args);
v8::Handle<v8::Value> createPattern(const v8::Arguments& args);

//Paths
v8::Handle<v8::Value> beginPath(const v8::Arguments& args);
v8::Handle<v8::Value> closePath(const v8::Arguments& args);
v8::Handle<v8::Value> fill(const v8::Arguments& args);
v8::Handle<v8::Value> stroke(const v8::Arguments& args);
v8::Handle<v8::Value> clip(const v8::Arguments& args);

v8::Handle<v8::Value> moveTo(const v8::Arguments& args);
v8::Handle<v8::Value> lineTo(const v8::Arguments& args);
v8::Handle<v8::Value> quadraticCurveTo(const v8::Arguments& args);
v8::Handle<v8::Value> bezierCurveTo(const v8::Arguments& args);
v8::Handle<v8::Value> arcTo(const v8::Arguments& args);
v8::Handle<v8::Value> arc(const v8::Arguments& args);
v8::Handle<v8::Value> rect(const v8::Arguments& args);
v8::Handle<v8::Value> isPointInPath(const v8::Arguments& args);

//Text
v8::Handle<v8::Value> getterFont(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterFont(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterTextAlign(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterTextAlign(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> getterTextBaseline(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterTextBaseline(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> fillText(const v8::Arguments& args);
v8::Handle<v8::Value> strokeText(const v8::Arguments& args);
v8::Handle<v8::Value> measureText(const v8::Arguments& args);

//Rectangles
v8::Handle<v8::Value> clearRect(const v8::Arguments& args);
v8::Handle<v8::Value> fillRect(const v8::Arguments& args);
v8::Handle<v8::Value> strokeRect(const v8::Arguments& args);

//New
v8::Handle<v8::Value> getterAntiAliasing(v8::Local<v8::String> property, const v8::AccessorInfo& info);
void setterAntiAliasing(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info);

v8::Handle<v8::Value> saveToFile(const v8::Arguments& args);