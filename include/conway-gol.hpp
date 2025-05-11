#pragma once

#include <ostream>
#include <memory>


class World
{

private:
	std::unique_ptr<bool[]> _mat;
	int _n = 0;
	int _m = 0;

public:
	World(int n, int m);
	~World();

	bool &operator()(int x, int y);
	const bool &operator()(int x, int y) const;
	World &operator=(const World &w);

	void clear();

	friend std::ostream &operator<<(std::ostream &os, const World &w);

	void init();
};
