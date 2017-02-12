
#pragma once

#include <iostream>

#define NOMINMAX
#include <Windows.h>

#define COUT_R(x) ( std::cout << caffepro::_red  	<< (x) << caffepro::_white )
#define COUT_Y(x) ( std::cout << caffepro::_yellow  << (x) << caffepro::_white )
#define COUT_B(x) ( std::cout << caffepro::_blue	<< (x) << caffepro::_white )
#define COUT_G(x) ( std::cout << caffepro::_green	<< (x) << caffepro::_white )

// GREEN - normal running
#define COUT_READ ( std::cout << caffepro::_green << "[ READ ]  " << caffepro::_white )
#define COUT_WRIT ( std::cout << caffepro::_green << "[ WRIT ]  " << caffepro::_white )
#define COUT_RUNN ( std::cout << caffepro::_green << "[ RUNN ]  " << caffepro::_white )
#define COUT_SUCC ( std::cout << caffepro::_green << "[ SUCC ]  " << caffepro::_white )

// YELLOW - warning
#define COUT_WARN ( std::cout << caffepro::_yellow << "[ WARN ]  " << caffepro::_white )
#define COUT_METD ( std::cout << caffepro::_yellow << "[ METD ]  " << caffepro::_white )

// BLUE - some display
#define COUT_WORKER(id, x) ( std::cout << caffepro::_blue << "[ ID " << (id) << " ]  " << (x) << caffepro::_white )
#define COUT_WORKID(id) ( std::cout << caffepro::_blue << "[ ID " << (id) << " ]  " << caffepro::_white )
#define COUT_CHEK ( std::cout << caffepro::_blue << "[ CHEK ]  " << caffepro::_white )

// RED - horrible mistake


namespace caffepro {

	inline std::ostream& _blue(std::ostream &s)
	{
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hStdout, FOREGROUND_BLUE
			| FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		return s;
	}

	inline std::ostream& _red(std::ostream &s)
	{
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hStdout,
			FOREGROUND_RED | FOREGROUND_INTENSITY);
		return s;
	}

	inline std::ostream& _green(std::ostream &s)
	{
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hStdout,
			FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		return s;
	}

	inline std::ostream& _yellow(std::ostream &s)
	{
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hStdout,
			FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);
		return s;
	}

	inline std::ostream& _white(std::ostream &s)
	{
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hStdout,
			FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
		return s;
	}

	struct color {
		color(WORD attribute) :m_color(attribute) {};
		WORD m_color;
	};

	template <class _Elem, class _Traits>
	std::basic_ostream<_Elem, _Traits>&
		operator<<(std::basic_ostream<_Elem, _Traits>& i, color& c)
	{
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hStdout, c.m_color);
		return i;
	}

}