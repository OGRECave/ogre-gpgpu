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

#include "CanvasV8Functions.h"

using namespace v8;

std::string toString(v8::Local<v8::Value> value)
{
	char buffer[256];
	Local<v8::String> s = value->ToString();
	s->WriteAscii((char*)&buffer);
	
	return std::string(buffer);
}

Ogre::Canvas::Logger* getCanvasLoggerPointer(const v8::Arguments& args)
{
	return static_cast<Ogre::Canvas::Logger*>(args.Holder()->GetPointerFromInternalField(0));
}

Ogre::Image* getImagePointer(const v8::Arguments& args)
{
	return static_cast<Ogre::Image*>(args.Holder()->GetPointerFromInternalField(0));
}

Ogre::Canvas::Gradient* getGradientPointer(const Arguments& args)
{
	return static_cast<Ogre::Canvas::Gradient*>(args.Holder()->GetPointerFromInternalField(0));
}

Ogre::Canvas::Context* getCanvasContextPointer(const Arguments& args)
{
	return static_cast<Ogre::Canvas::Context*>(args.Holder()->GetPointerFromInternalField(0));
}

Ogre::Canvas::Context* getCanvasContextPointer(const AccessorInfo& info)
{
	return static_cast<Ogre::Canvas::Context*>(info.Holder()->GetPointerFromInternalField(0));
}

Ogre::Canvas::V8Context* getV8CanvasContext(const Arguments& args)
{
	return static_cast<Ogre::Canvas::V8Context*>(args.Holder()->GetPointerFromInternalField(1));
}

Ogre::ColourValue getColor(Local<Value> value)
{
	Ogre::ColourValue color = Ogre::ColourValue::Black;
	std::string text = toString(value);

	if (text[0] == '#')
		color = Ogre::Canvas::ColourConverter::fromHexa(text);
	else if (text.substr(0, 4) == "rgba")
	{
		std::string tmp = text.substr(5, text.size()-6);
		Ogre::StringVector numbers = Ogre::StringUtil::split(tmp, ",");
		color.r = (float) atof(numbers[0].c_str()) / 255.0f;
		color.g = (float) atof(numbers[1].c_str()) / 255.0f;
		color.b = (float) atof(numbers[2].c_str()) / 255.0f;
		color.a = (float) atof(numbers[3].c_str());
	}
	else if (text.substr(0, 3) == "rgb")
	{
		std::string tmp = text.substr(4, text.size()-5);
		Ogre::StringVector numbers = Ogre::StringUtil::split(tmp, ",");
		color.r = (float) atof(numbers[0].c_str()) / 255.0f;
		color.g = (float) atof(numbers[1].c_str()) / 255.0f;
		color.b = (float) atof(numbers[2].c_str()) / 255.0f;
	}
	else if (text == "white")
		color = Ogre::ColourValue::White;
	else if (text == "black")
		color = Ogre::ColourValue::Black;
	else if (text == "red")
		color = Ogre::ColourValue::Red;
	else if (text == "green")
		color = Ogre::ColourValue::Green;
	else if (text == "blue")
		color = Ogre::ColourValue::Blue;
	else if (text == "yellow")
		color = Ogre::ColourValue(1.0, 1.0, 0.0);

	return color;
}

Ogre::Canvas::Gradient* getGradient(v8::Local<v8::Value> value)
{
	return static_cast<Ogre::Canvas::Gradient*>(value->ToObject()->GetPointerFromInternalField(0));
}

Ogre::Canvas::Pattern* getPattern(v8::Local<v8::Value> value)
{
	return static_cast<Ogre::Canvas::Pattern*>(value->ToObject()->GetPointerFromInternalField(0));
}

Ogre::Image* getImage(v8::Local<v8::Value> value)
{
	return static_cast<Ogre::Image*>(value->ToObject()->GetPointerFromInternalField(0));
}

/**************************
 *  		Console		  *
 **************************/
Handle<Value> log(const Arguments& args)
{
	std::string message = toString(args[0]);
	getCanvasLoggerPointer(args)->log(message);

	return v8::True();
}

/**************************
 *  		Image		  *
 **************************/
Handle<Value> loadImage(const Arguments& args)
{
	std::string filename = toString(args[0]);

	Ogre::Image* image = new Ogre::Image;
	image->load(filename, "General");
	
	Ogre::Canvas::V8Context* v8ctx = getV8CanvasContext(args);
	Handle<Function> imageConstructor = v8ctx->mImageTemplate->GetFunction();
	Local<Object> obj = imageConstructor->NewInstance();
	obj->SetPointerInInternalField(0, image);

	return obj;	
}

/**********************************
 *  		Canvas Gradient		  *
 **********************************/
Handle<Value> addColorStop(const Arguments& args)
{
	float offset            = (float) args[0]->NumberValue();
	Ogre::ColourValue color = getColor(args[1]);

	getGradientPointer(args)->addColorStop(offset, color);

	return v8::True();
}

Handle<Value> gradient_destroy(const Arguments& args)
{
	Ogre::Canvas::Gradient* gradient = getGradientPointer(args);
	delete gradient;

	return v8::True();
}

/******************************
 *  		2D Context		  *
 ******************************/
Handle<Value> save(const Arguments& args)
{
	getCanvasContextPointer(args)->save();

	return v8::True();
}

Handle<Value> restore(const Arguments& args)
{
	getCanvasContextPointer(args)->restore();

	return v8::True();
}

/**********************************
 *  		Transformation		  *
 **********************************/
Handle<Value> scale(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();
	getCanvasContextPointer(args)->scale(x, y);

	return v8::True();
}

Handle<Value> rotate(const Arguments& args)
{
	float angleRadian = (float) args[0]->NumberValue();
	getCanvasContextPointer(args)->rotate(angleRadian);

	return v8::True();
}

Handle<Value> translate(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();
	getCanvasContextPointer(args)->translate(x, y);

	return v8::True();
}

Handle<Value> transform(const Arguments& args)
{
	float m11 = (float) args[0]->NumberValue();
	float m12 = (float) args[1]->NumberValue();
	float m21 = (float) args[2]->NumberValue();
	float m22 = (float) args[3]->NumberValue();
	float dx  = (float) args[4]->NumberValue();
	float dy  = (float) args[5]->NumberValue();
	getCanvasContextPointer(args)->transform(m11, m12, m21, m22, dx, dy);

	return v8::True();
}

Handle<Value> setTransform(const Arguments& args)
{
	float m11 = (float) args[0]->NumberValue();
	float m12 = (float) args[1]->NumberValue();
	float m21 = (float) args[2]->NumberValue();
	float m22 = (float) args[3]->NumberValue();
	float dx  = (float) args[4]->NumberValue();
	float dy  = (float) args[5]->NumberValue();
	getCanvasContextPointer(args)->setTransform(m11, m12, m21, m22, dx, dy);

	return v8::True();
}
/**********************************
 *  		Image drawing		  *
 **********************************/
Handle<Value> drawImage(const Arguments& args)
{
	Ogre::Canvas::Context* ctx  = getCanvasContextPointer(args);
	Ogre::Image* image = getImage(args[0]);

	if (args.Length() == 3)
	{
		float x = (float) args[1]->NumberValue();
		float y = (float) args[2]->NumberValue();
		ctx->drawImage(*image, x, y);	
	}
	else if (args.Length() == 5)
	{
		float x = (float) args[1]->NumberValue();
		float y = (float) args[2]->NumberValue();
		float w = (float) args[3]->NumberValue();
		float h = (float) args[4]->NumberValue();
		ctx->drawImage(*image, x, y, w, h);
	}
	else if (args.Length() == 9)
	{
		float srcX      = (float) args[1]->NumberValue();
		float srcY      = (float) args[2]->NumberValue();
		float srcWidth  = (float) args[3]->NumberValue();
		float srcHeight = (float) args[4]->NumberValue();
		float dstX      = (float) args[5]->NumberValue();
		float dstY      = (float) args[6]->NumberValue();
		float dstWidth  = (float) args[7]->NumberValue();
		float dstHeight = (float) args[8]->NumberValue();
		ctx->drawImage(*image, srcX, srcY, srcWidth, srcHeight, dstX, dstY, dstWidth, dstHeight);
	}

	return v8::True();
}


/******************************
 *  		Compositing		  *
 ******************************/
Handle<Value> getterGlobalAlpha(Local<v8::String> property, const AccessorInfo& info)
{
	float alpha = getCanvasContextPointer(info)->globalAlpha();
	return Number::New(alpha);
}

void setterGlobalAlpha(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	float alpha = (float) value->NumberValue();
	getCanvasContextPointer(info)->globalAlpha(alpha);
}

Handle<Value> getterGlobalCompositeOperation(Local<String> property, const AccessorInfo& info)
{
	
	Ogre::Canvas::DrawingOperator op = getCanvasContextPointer(info)->globalCompositeOperation();

	if (op == Ogre::Canvas::DrawingOperator_Copy)
		return v8::String::New("copy");
	else if (op == Ogre::Canvas::DrawingOperator_SourceOver)
		return v8::String::New("source-over");
	else if (op == Ogre::Canvas::DrawingOperator_SourceIn)
		return v8::String::New("source-in");
	else if (op == Ogre::Canvas::DrawingOperator_SourceOut)
		return v8::String::New("source-out");
	else if (op == Ogre::Canvas::DrawingOperator_SourceATop)
		return v8::String::New("source-atop");
	else if (op == Ogre::Canvas::DrawingOperator_DestOver)
		return v8::String::New("destination-over");
	else if (op == Ogre::Canvas::DrawingOperator_DestIn)
		return v8::String::New("destination-in");
	else if (op == Ogre::Canvas::DrawingOperator_DestOut)
		return v8::String::New("destination-out");
	else if (op == Ogre::Canvas::DrawingOperator_DestATop)
		return v8::String::New("destination-atop");
	else if (op == Ogre::Canvas::DrawingOperator_Xor)
		return v8::String::New("xor");
	else if (op == Ogre::Canvas::DrawingOperator_PlusDarker)
		return v8::String::New("plus-darker");
	else if (op == Ogre::Canvas::DrawingOperator_Highlight)
		return v8::String::New("highlight");
	else if (op == Ogre::Canvas::DrawingOperator_PlusLighter)
		return v8::String::New("plus-lighter");
	else //if (op == Ogre::Canvas::DrawingOperator_Clear)
		return v8::String::New("clear");
}

void setterGlobalCompositeOperation(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	Ogre::Canvas::DrawingOperator op = Ogre::Canvas::DrawingOperator_SourceOver;
	std::string text = toString(value);

	if (text == "copy")
		op = Ogre::Canvas::DrawingOperator_Copy;
	else if (text == "source-over")
		op = Ogre::Canvas::DrawingOperator_SourceOver;
	else if (text == "source-in")
		op = Ogre::Canvas::DrawingOperator_SourceIn;
	else if (text == "source-out")
		op = Ogre::Canvas::DrawingOperator_SourceOut;
	else if (text == "source-atop")
		op = Ogre::Canvas::DrawingOperator_SourceATop;
	else if (text == "destination-over")
		op = Ogre::Canvas::DrawingOperator_DestOver;
	else if (text == "destination-in")
		op = Ogre::Canvas::DrawingOperator_DestIn;
	else if (text == "destination-out")
		op = Ogre::Canvas::DrawingOperator_DestOut;
	else if (text == "destination-atop")
		op = Ogre::Canvas::DrawingOperator_DestATop;
	else if (text == "xor")
		op = Ogre::Canvas::DrawingOperator_Xor;
	else if (text == "plus-darker")
		op = Ogre::Canvas::DrawingOperator_PlusDarker;
	else if (text == "highlight")
		op = Ogre::Canvas::DrawingOperator_Highlight;
	else if (text == "plus-lighter")
		op = Ogre::Canvas::DrawingOperator_PlusLighter;
	else if (text == "clear")
		op = Ogre::Canvas::DrawingOperator_Clear;

	getCanvasContextPointer(info)->globalCompositeOperation(op);
}

/******************************
 *  		Line styles		  *
 ******************************/
Handle<Value> getterLineWidth(Local<String> property, const AccessorInfo& info)
{
	float width = getCanvasContextPointer(info)->lineWidth();
	return Number::New(width);
}
void setterLineWidth(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	float width = (float) value->NumberValue();
	getCanvasContextPointer(info)->lineWidth(width);
}

Handle<Value> getterLineCap(Local<String> property, const AccessorInfo& info)
{
	Ogre::Canvas::LineCap lineCap = getCanvasContextPointer(info)->lineCap();

	if (lineCap == Ogre::Canvas::LineCap_Butt)
		return v8::String::New("butt");
	else if (lineCap == Ogre::Canvas::LineCap_Round)
		return v8::String::New("round");
	else //if (lineCap == Ogre::Canvas::LineCap_Square)
		return v8::String::New("square");
}

void setterLineCap(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	Ogre::Canvas::LineCap lineCap = Ogre::Canvas::LineCap_Butt;
	std::string text = toString(value);

	if (text == "round")
		lineCap = Ogre::Canvas::LineCap_Round;
	else if (text == "square")
		lineCap = Ogre::Canvas::LineCap_Square;

	getCanvasContextPointer(info)->lineCap(lineCap);
}

Handle<Value> getterLineJoin(Local<String> property, const AccessorInfo& info)
{
	Ogre::Canvas::LineJoin lineJoin = getCanvasContextPointer(info)->lineJoin();

	if (lineJoin == Ogre::Canvas::LineJoin_Miter)
		return v8::String::New("miter");
	else if (lineJoin == Ogre::Canvas::LineJoin_Round)
		return v8::String::New("round");
	else //if (lineJoin == Ogre::Canvas::LineJoin_Bevel)
		return v8::String::New("bevel");
}

void setterLineJoin(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	Ogre::Canvas::LineJoin lineJoin = Ogre::Canvas::LineJoin_Miter;
	std::string text = toString(value);

	if (text == "round")
		lineJoin = Ogre::Canvas::LineJoin_Round;
	else if (text == "bevel")
		lineJoin = Ogre::Canvas::LineJoin_Bevel;

	getCanvasContextPointer(info)->lineJoin(lineJoin);
}

Handle<Value> getterMiterLimit(Local<String> property, const AccessorInfo& info)
{
	float mitterLimit = getCanvasContextPointer(info)->miterLimit();
	return Number::New(mitterLimit);
}

void setterMiterLimit(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	float limit = (float) value->NumberValue();
	getCanvasContextPointer(info)->miterLimit(limit);
}

Handle<Value> getterLineDash(Local<String> property, const AccessorInfo& info)
{
	Ogre::Canvas::LineDash lineDash = getCanvasContextPointer(info)->lineDash();

	if (lineDash == Ogre::Canvas::LineDash_Dashed)
		return v8::String::New("dashed");
	else if (lineDash == Ogre::Canvas::LineDash_Dotted)
		return v8::String::New("dotted");
	else //if (lineDash == Ogre::Canvas::LineDash_Solid)
		return v8::String::New("solid");
}

void setterLineDash(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	Ogre::Canvas::LineDash lineDash = Ogre::Canvas::LineDash_Solid;
	std::string text = toString(value);

	if (text == "dotted")
		lineDash = Ogre::Canvas::LineDash_Dotted;
	else if (text == "dashed")
		lineDash = Ogre::Canvas::LineDash_Dashed;

	getCanvasContextPointer(info)->lineDash(lineDash);
}

/**********************************************
 *  		Colors, styles and shadows		  *
 **********************************************/
Handle<Value> getterStrokeStyle(Local<String> property, const AccessorInfo& info)
{
	return v8::True();
}

void setterStrokeStyle(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	Ogre::Canvas::Context* ctx = getCanvasContextPointer(info);

	if (value->IsObject())
	{
		Ogre::Canvas::Gradient* gradient = getGradient(value);
		if (gradient != NULL)
			ctx->strokeStyle(gradient);
		else
			ctx->strokeStyle(Ogre::ColourValue::Black);
	}
	else 
	{
		Ogre::ColourValue color = getColor(value);
		ctx->strokeStyle(color);
	}
}

Handle<Value> getterFillStyle(Local<String> property, const AccessorInfo& info)
{
	return v8::True();
}

void setterFillStyle(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	Ogre::Canvas::Context* ctx = getCanvasContextPointer(info);

	if (value->IsObject())
	{
		std::string str = toString(value);
		if (str == "[object CanvasGradient]") //beurk !
		{
			Ogre::Canvas::Gradient* gradient = getGradient(value);
			ctx->fillStyle(gradient);
		}
		else if (str == "[object CanvasPattern]")
		{
			Ogre::Canvas::Pattern* pattern = getPattern(value);
			ctx->fillStyle(pattern);
		}
	}
	else 
	{
		Ogre::ColourValue color = getColor(value);
		ctx->fillStyle(color);
	}
}

Handle<Value> getterShadowOffsetX(Local<String> property, const AccessorInfo& info)
{
	return v8::True();
}
void setterShadowOffsetX(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	float x = (float) value->NumberValue();
	getCanvasContextPointer(info)->shadowOffsetX(x);
}

Handle<Value> getterShadowOffsetY(Local<String> property, const AccessorInfo& info)
{
	return v8::True();
}

void setterShadowOffsetY(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	float y = (float) value->NumberValue();
	getCanvasContextPointer(info)->shadowOffsetY(y);
}

Handle<Value> getterShadowBlur(Local<String> property, const AccessorInfo& info)
{
	return v8::True();
}

void setterShadowBlur(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	float blur = (float) value->NumberValue();
	getCanvasContextPointer(info)->shadowBlur(blur);
}

Handle<Value> getterShadowColor(Local<String> property, const AccessorInfo& info)
{
	return v8::True();
}

void setterShadowColor(Local<String> property, Local<Value> value, const AccessorInfo& info)
{
	Ogre::ColourValue color = getColor(value);
	getCanvasContextPointer(info)->shadowColor(color);
}


Handle<Value> createLinearGradient(const Arguments& args)
{
	float x0 = (float) args[0]->NumberValue();
	float y0 = (float) args[1]->NumberValue();	
	float x1 = (float) args[2]->NumberValue();
	float y1 = (float) args[3]->NumberValue();

	Ogre::Canvas::Gradient* gradient = getCanvasContextPointer(args)->createLinearGradient(x0, y0, x1, y1);

	Handle<Function> canvasGradientConstructor = getV8CanvasContext(args)->mCanvasGradientTemplate->GetFunction();
	Local<Object> obj = canvasGradientConstructor->NewInstance();
	obj->SetPointerInInternalField(0, gradient);

	return obj;
}

Handle<Value> createRadialGradient(const Arguments& args)
{
	float x0 = (float) args[0]->NumberValue();
	float y0 = (float) args[1]->NumberValue();	
	float r0 = (float) args[2]->NumberValue();	
	float x1 = (float) args[3]->NumberValue();
	float y1 = (float) args[4]->NumberValue();
	float r1 = (float) args[5]->NumberValue();	

	Ogre::Canvas::Gradient* gradient = getCanvasContextPointer(args)->createRadialGradient(x0, y0, r0, x1, y1, r1);
	
	Handle<Function> canvasGradientConstructor = getV8CanvasContext(args)->mCanvasGradientTemplate->GetFunction();
	Local<Object> obj = canvasGradientConstructor->NewInstance();
	obj->SetPointerInInternalField(0, gradient);

	return obj;
}

Handle<Value> createPattern(const Arguments& args)
{
	Ogre::Image* image = getImage(args[0]);
	std::string repeatMode = toString(args[1]);

	Ogre::Canvas::Repetition repeat = Ogre::Canvas::Repetition_Repeat;
	Ogre::Canvas::Pattern* pattern = getCanvasContextPointer(args)->createPattern(*image, repeat);

	Handle<Function> canvasPatternConstructor = getV8CanvasContext(args)->mCanvasPatternTemplate->GetFunction();
	Local<Object> obj = canvasPatternConstructor->NewInstance();
	obj->SetPointerInInternalField(0, pattern);

	return obj;
}

/**************************
 *  		Paths		  *
 **************************/
Handle<Value> beginPath(const Arguments& args)
{
	getCanvasContextPointer(args)->beginPath();
	
	return v8::True();
}

Handle<Value> closePath(const Arguments& args)
{
	getCanvasContextPointer(args)->closePath();
	
	return v8::True();
}

Handle<Value> fill(const Arguments& args)
{
	getCanvasContextPointer(args)->fill();

	return v8::True();
}

Handle<Value> stroke(const Arguments& args)
{
	getCanvasContextPointer(args)->stroke();

	return v8::True();
}

Handle<Value> clip(const Arguments& args)
{
	getCanvasContextPointer(args)->clip();

	return v8::True();
}

Handle<Value> moveTo(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();	
	getCanvasContextPointer(args)->moveTo(x, y);

	return v8::True();
}

Handle<Value> lineTo(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();	
	getCanvasContextPointer(args)->lineTo(x, y);

	return v8::True();
}

Handle<Value> quadraticCurveTo(const Arguments& args)
{
	float cpx = (float) args[0]->NumberValue();
	float cpy = (float) args[1]->NumberValue();
	float x   = (float) args[2]->NumberValue();
	float y   = (float) args[3]->NumberValue();
	getCanvasContextPointer(args)->quadraticCurveTo(cpx, cpy, x, y);

	return v8::True();
}

Handle<Value> bezierCurveTo(const Arguments& args)
{
	float cp1x = (float) args[0]->NumberValue();
	float cp1y = (float) args[1]->NumberValue();	
	float cp2x = (float) args[2]->NumberValue();
	float cp2y = (float) args[3]->NumberValue();	
	float x    = (float) args[4]->NumberValue();
	float y    = (float) args[5]->NumberValue();	
	getCanvasContextPointer(args)->bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y);

	return v8::True();
}

Handle<Value> arcTo(const Arguments& args)
{
	float x1     = (float) args[0]->NumberValue();
	float y1     = (float) args[1]->NumberValue();
	float x2     = (float) args[2]->NumberValue();
	float y2     = (float) args[3]->NumberValue();
	float radius = (float) args[4]->NumberValue();

	getCanvasContextPointer(args)->arcTo(x1, y1, x2, y2, radius);

	return v8::True();
}

Handle<Value> arc(const Arguments& args)
{
	float x          = (float) args[0]->NumberValue();
	float y          = (float) args[1]->NumberValue();
	float radius     = (float) args[2]->NumberValue();
	float startAngle = (float) args[3]->NumberValue();
	float endAngle   = (float) args[4]->NumberValue();
	bool anticlockwise    = args[5]->BooleanValue();

	getCanvasContextPointer(args)->arc(x, y, radius, startAngle, endAngle, anticlockwise);

	return v8::True();
}

Handle<Value> rect(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();
	float w = (float) args[2]->NumberValue();
	float h = (float) args[3]->NumberValue();

	getCanvasContextPointer(args)->rect(x, y, w, h);

	return v8::True();
}

Handle<Value> isPointInPath(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();

	return Boolean::New(getCanvasContextPointer(args)->isPointInPath(x, y));
}

/**************************
 *  		Text		  *
 **************************/
v8::Handle<v8::Value> getterFont(v8::Local<v8::String> property, const v8::AccessorInfo& info)
{
	return v8::True();
}

void setterFont(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info)
{

}

v8::Handle<v8::Value> getterTextAlign(v8::Local<v8::String> property, const v8::AccessorInfo& info)
{
	return v8::True();
}

void setterTextAlign(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info)
{

}

v8::Handle<v8::Value> getterTextBaseline(v8::Local<v8::String> property, const v8::AccessorInfo& info)
{
	return v8::True();
}

void setterTextBaseline(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info)
{

}

Handle<Value> fillText(const Arguments& args)
{
	float maxWidth = -1;
	std::string text = toString(args[0]);
	float x = (float) args[1]->NumberValue();
	float y = (float) args[2]->NumberValue();
	if (args.Length() == 4)
		maxWidth = (float) args[3]->NumberValue();

	getCanvasContextPointer(args)->fillText(text, x, y, maxWidth);

	return v8::True();
}

Handle<Value> strokeText(const Arguments& args)
{
	float maxWidth = -1;
	std::string text = toString(args[0]);
	float x = (float) args[1]->NumberValue();
	float y = (float) args[2]->NumberValue();
	if (args.Length() == 4)
		maxWidth = (float) args[3]->NumberValue();

	getCanvasContextPointer(args)->strokeText(text, x, y, maxWidth);

	return v8::True();
}

Handle<Value> measureText(const Arguments& args)
{
	//TODO : not yet implemented
	return v8::True();
}

/******************************
 *  		Rectangles		  *
 ******************************/
Handle<Value> clearRect(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();
	float w = (float) args[2]->NumberValue();
	float h = (float) args[3]->NumberValue();

	getCanvasContextPointer(args)->clearRect(x, y, w, h);

	return v8::True();
}

Handle<Value> fillRect(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();
	float w = (float) args[2]->NumberValue();
	float h = (float) args[3]->NumberValue();

	getCanvasContextPointer(args)->fillRect(x, y, w, h);

	return v8::True();
}

Handle<Value> strokeRect(const Arguments& args)
{
	float x = (float) args[0]->NumberValue();
	float y = (float) args[1]->NumberValue();
	float w = (float) args[2]->NumberValue();
	float h = (float) args[3]->NumberValue();

	getCanvasContextPointer(args)->strokeRect(x, y, w, h);

	return v8::True();
}

/**********************
 *  		New		  *
 **********************/
v8::Handle<v8::Value> getterAntiAliasing(v8::Local<v8::String> property, const v8::AccessorInfo& info)
{
	return v8::True();
}

void setterAntiAliasing(v8::Local<v8::String> property, v8::Local<v8::Value> value, const v8::AccessorInfo& info)
{

}

Handle<Value> saveToFile(const Arguments& args)
{
	std::string filename = toString(args[0]);
	getCanvasContextPointer(args)->saveToFile(filename);

	return v8::True();
}