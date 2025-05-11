#include "conway-gol.hpp"

#include <iostream>
#include <memory>
#include <vector>

World::World(int n, int m) : _n(n), _m(m)
{
	_mat = std::make_unique<bool[]>(n * m);
}

World::~World(){}

bool &World::operator()(int x, int y)
{
	x = (x + _n) % _n;
	y = (y + _m) % _m;
	return _mat[x + y * _n];
}

const bool &World::operator()(int x, int y) const
{
	x = (x + _n) % _n;
	y = (y + _m) % _m;
	return _mat[x + y * _n];
}

World &World::operator=(const World &w)
{
	for (int i = 0; i < _n; i++)
		for (int j = 0; j < _m; j++)
			(*this)(i, j) = w(i, j);
	return *this;
}

void World::clear() {
	// clears the matrix, duh
	_mat = std::make_unique<bool[]>(_n * _m);
}

std::ostream &operator<<(std::ostream &os, const World &w)
{
	for (int i = 0; i < w._n; i++)
	{
		for (int j = 0; j < w._m; j++)
			os << (w(i, j) ? "o" : ".") << " ";
		os << std::endl;
	}
	return os;
}


void World::init() {
	const std::vector<std::pair<int,int>> gosper_glider_gun = {
		{24,0},{22,1},{24,1},
		{12,2},{13,2},{20,2},{21,2},{34,2},{35,2},
		{11,3},{15,3},{20,3},{21,3},{34,3},{35,3},
		{0,4},{1,4},{10,4},{11,4},{15,4},{16,4},{22,4},{24,4},
		{0,5},{1,5},{10,5},{12,5},{16,5},{17,5},{22,5},{24,5},
		{10,6},{11,6},{12,6},{13,6},{14,6},{15,6},{16,6},
		{0,7},{1,7},{2,7},{3,7},{4,7},{5,7},{6,7},{7,7},{8,7},{9,7},
		{10,7},{11,7},{12,7},{13,7},{14,7},{15,7},{16,7}
	};

	for (auto [i, j] : gosper_glider_gun)
		(*this)(i, j) = true;
}