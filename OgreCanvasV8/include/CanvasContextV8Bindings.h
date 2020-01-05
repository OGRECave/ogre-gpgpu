#pragma once

#include "CanvasContext.h"
#include "CanvasLogger.h"

#include "v8.h"

class CanvasContextV8Bindings
{
	public:
		CanvasContextV8Bindings();
		void loadScript(const std::string& _filename, OgreCanvas::CanvasContext* _canvasContext, OgreCanvas::CanvasLogger* _console);
		void executeJS(const std::string& _js);
		void dispose();
		static std::string readScript(const std::string& _filename);

		static OgreCanvas::CanvasContext*        context2D;
		static v8::Persistent<v8::Context>       contextV8;
		static v8::Persistent<v8::FunctionTemplate>  canvasGradientTemplate;
		static v8::Persistent<v8::FunctionTemplate>  canvasPatternTemplate;
		static v8::Persistent<v8::FunctionTemplate>  imageTemplate;
		static v8::Persistent<v8::FunctionTemplate>  consoleTemplate;
};