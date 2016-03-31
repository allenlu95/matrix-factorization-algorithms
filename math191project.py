import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# 
# Algorithm 4.5 in the paper
#
# Computes an orthogonal matrix to approximate the range of the input matrix
#
# @param A: the input matrix
# @param k: desired integer rank approximation
# @param oversampling: integer oversampling parameter for better rank approximation
# @param givens: boolean parameter to choose from subsampled random Fourier transform 
#		and modified subsampled random Fourier transform with Givens
# @param testing: boolean parameter for testing purposes
# @return: an orthogonal m x (k + oversamling) matrix which approximates
#		the range of A
#
def fast_randomized_range_finder(A, k, oversampling, givens=False, testing=False):
	l = k + oversampling
	m, n = A.shape
	
	if givens is False:
		if testing:
			print "subsampled_random_fourier_transform"
		Omega = subsampled_random_fourier_transform(n, l)
	else:
		if testing:
			print "givens_subsampled_random_fourier_transform"
		Omega = givens_subsampled_random_fourier_transform(n, l)
	
	Y = np.dot(A, Omega)

	Q, R = la.qr(Y)
	return Q


#
# @param n: the number of rows of the matrix
# @param l: the number of columns of the matrix
# @return: an nxl subsampled random Fourier transform matrix
#
def subsampled_random_fourier_transform(n, l):
	D = random_complex_diagonal(n)
	F = unitary_discrete_fourier_transform(n)
	R = random_permutation_matrix(n, l)
	Omega = np.sqrt(n/l) * np.dot(np.dot(D, F), R)
	return Omega


#
# @param n: the size of the matrix
# @return: nxn diagonal matrix whose entries are independent random variables
#
def random_complex_diagonal(n):
	A = np.random.normal(size=(n,)) + 1j * np.random.normal(size=(n,))
	D = np.diag(A/la.norm(A))
	return D


#
# @param n: the integer size of the matrix
# @return: nxn unitary discrete Fourier transform matrix
#
def unitary_discrete_fourier_transform(n):
	P = np.arange(0, n).repeat(n).reshape((n, n))
	F = np.sqrt(n) * np.exp(-2 * np.pi * 1j * P * P.T / n)
	return F


#
# @param n: the integer number of rows
# @param l: the integer number of columns
# @return: a matrix that samples its columns uniformly from the identity matrix
#
def random_permutation_matrix(n, l):
	R = np.identity(n)
	np.random.shuffle(R)
	R = R[:, :l]
	return R


#
# @param n: the integer number of rows
# @param l: the integer number of columns
# @return: the subsampled random Fourier transform matrix using Givens Rotations
#
def givens_subsampled_random_fourier_transform(n, l):
	D = random_complex_diagonal(n)
	D_Prime = random_complex_diagonal(n)
	D_Prime_Prime = random_complex_diagonal(n)

	F = unitary_discrete_fourier_transform(n)
	R = random_permutation_matrix(n, l)

	Theta = random_givens_rotation_product(n)
	Theta_Prime = random_givens_rotation_product(n)

	# Omega = Dprimeprime*Thetaprime*Dprime*Theta*D*F*R
	half_of_Omega = np.dot(np.dot(np.dot(D_Prime_Prime, Theta_Prime), D_Prime), Theta)
	Omega = np.dot(np.dot(np.dot(half_of_Omega, D), F), R)

	return Omega


#
# @param n: the integer size of the matrix
# @return: product of random Givens matrices applied to each cell one above 
#		the diagonal of a permutation of identity matrix
#
def random_givens_rotation_product(n):
	P = random_permutation_matrix(n, n)
	for i in xrange(n-1):
		P = np.dot(P, givens_rotation(n, i, i+1, np.random.normal()))
	return P


#
# Computes the Givens Rotation matrix
#
# @param n: the integer size of the matrix
# @param i: the integer row index
# @param j: the integer column index
# @param theta: the angle of rotation
# @return: Givens Rotation matrix of size n with row, column indices i, j 
#		and angle theta
#
def givens_rotation(n, i, j, theta):
	givens = np.identity(n)
	givens[i, i] = np.cos(theta)
	givens[j, j] = givens[i, i]
	givens[i, j] = np.sin(theta)
	givens[j, i] = -givens[i, j]
	return givens


# 
# Algorithm 4.1 and 4.3 in the paper: it's 4.1 when q == 0 and 4.5 when q > 0
#
# Computes an orthogonal matrix to approximate the range of the input matrix
#
# @param A: the input matrix
# @param k: desired integer rank approximation
# @param oversampling: integer oversampling parameter for better rank approximation
# @param q: an integer for the number of power iterations (AA^*)^q
# @param orth: optional boolean parameter with default value False. Orthogonalizes 
# 		Y at each iteration if set to True; does nothing otherwise
# @return: an orthogonal m x (k + oversamling) matrix which approximates
#		the range of A
#
def randomized_power_iteration(A, k, oversampling, q, orth=False):
	m, n = A.shape
	l = k + oversampling

	# Gaussian random matrix Omega
	Omega = np.random.randn(n, l)
	Y = np.dot(A, Omega)

	# The power iteration
	i = 1
	while i <= q:
		T = np.dot(A.T, Y)
		Y = np.dot(A, T)
		if orth:
			Y, G = la.qr(Y)
		i += 1

	# Constructing an m x l matrix Q whose columns form an orthonormal
	# basis for the range of Y using QR factorization
	Q, R = la.qr(Y)
	return Q


#
# Computes the Stage B SVD of randomized algorithm
#
# @param A: the input matrix
# @param Q: the orthogonal matrix which approxmates the range of A
#
def stage_b_SVD(A, Q):
	B = np.dot(Q.T, A)
	U, S, V = la.svd(B, full_matrices=False)
	U = np.dot(Q, U)

	return (U, S, V)

#
# Computes the SVD of the input matrix using randomized power iteration algorithm
#
# @param A: the input parameter
# @param k: the wanted integer rank approximation
# @param oversampling: integer oversampling parameter for better rank approximation
# @param q: an integer for the number of power iterations (AA^*)^q
# @return: rank k approximation SVD of the matrix A as (U, S, V)
#
def random_SVD_iterated(A, k, oversampling, q):
	# Stage A
	Q = randomized_power_iteration(A, k, oversampling, q)

	# Stage B
	return stage_b_SVD(A, Q)


#
# Computes the SVD of the input matrix using fast randomized range finder
#
# @param A: the input parameter
# @param k: the wanted integer rank approximation
# @param oversampling: integer oversampling parameter for better rank approximation
# @return: rank k approximation SVD of the matrix A as (U, S, V)
#
def random_SVD_fast(A, k, oversampling):
	# Stage A
	Q = fast_randomized_range_finder(A, k, oversampling)

	# Stage B
	return stage_b_SVD(A, Q)



def random_SVD_test(k, q):
	m = 100000
	n = 200

	U0, R = la.qr(np.random.randn(m, n))
	V, R = la.qr(np.random.randn(n, n))
	U = U0[:, 0:n]

	S = np.ones((n))
	t = 2.0

	j = 1
	while j < n:
		S[j] = 1/t
		t *= 2
		j += 1

	A = np.dot(U, np.dot(np.diag(S), V))

	U, S, V = random_SVD_iterated(A, k, k, q)
	dNormSigma = (la.norm(A-np.dot(U, np.dot(np.diag(S), V))), S[k+1])
	print dNormSigma

	U, S, V = random_SVD_fast(A, k, k)
	dNormSigma = (la.norm(A-np.dot(U, np.dot(np.diag(S), V))), S[k+1])
	print dNormSigma


def main():
	random_SVD_test(10, 10)

if __name__ == "__main__":
	main()

















