import pytest

from kmeans import calculate_centers, generate_k, assign_points, update_centers


class TestKMeans(object):

    def pytest_funcarg__centers(self, request):
        return [[.67, 2.0], [10.0, 8.0]]

    def test_point_avg(self):
        two_dimensional_points = (0, 10), (5, 15)
        three_dimensional_points = (1, 2, 6), (2, 3, 8), (4, 5, 3), (6, 7, 0)
        assert calculate_centers(two_dimensional_points) == [2.5, 12.5]
        assert calculate_centers(three_dimensional_points) == [3.25, 4.25, 4.25]

    def test_generate_k_two_dimensions(self):
        two_dimensional_points = (0, 10), (5, 15)
        for i in range(1000):
            centers = generate_k(two_dimensional_points, 10)
            for point in centers:
                assert 0 < point[0] < 10
                assert 5 < point[1] < 15

    def test_generate_k_three_dimensions(self):
        three_dimensional_points = (1, 2, 6), (2, 3, 8), (4, 5, 3), (6, 7, 0)
        for i in range(1000):
            centers = generate_k(three_dimensional_points, 10)
            for point in centers:
                assert 1 < point[0] < 6
                assert 2 < point[1] < 7
                assert 0 < point[1] < 8

    def test_update_centers(self):
        data_set = (0, 1), (1, 2), (1, 3), (10, 9), (11, 8), (9, 7)
        assignments = [0, 0, 0, 1, 1, 1]

        centers = update_centers(data_set, assignments)
        assert len(centers) == 2

        rounded_centers = []
        for point in centers:
            rounded_centers.append([float('%.2f' % point[0]), float('%.2f' % point[1])])
        for point in [[.67, 2.0], [10.0, 8.0]]:
            assert point in rounded_centers

    def test_assign_points(self):
        data_set = (0, 1), (1, 2), (1, 3), (10, 9), (11, 8), (9, 7)
        assignments = [0, 0, 0, 1, 1, 1]
        centers = [[.67, 2.0], [10.0, 8.0]]
        assert assign_points(data_set, centers) == assignments