import numpy as np 
import timeit 
import random 
from scipy.linalg import lu_factor , lu_solve 
import scipy 
import time 
import matplotlib.pyplot as plt

def test_lu(n): 

	#  initialisation des axes 
	y_scipy= [] 
	y_notre = []
	y_tridiag=[]  

	# initialisation de l'axe temps 
	x = []

	while n<100: 

		x.append(n) 
		
		# initialisation des diagonales de la matrice tridiagonale 
		a = [1 for i in range(n-1)]; b = [-2 for i in range(n)]; c = [1 for i in range(n-1)] 

		# initialiser la matrice tridiagonale A 
		A = tridiag(a, b, c) 

		# initialiser le vecteur u 
		u=c  

		# initiaiser le vecteur l 
		l=[0 for i in range(n)] 

		# initialiser le vecteur v 
		v= [0 for i in range(n)]  

		# v1 = a1 
		v[0] = A[0,0] 


		# start profiling of lu_tridiag 
		start_time2 = time.time() 
		res2= lu_tridiag(A,l,u,v,a,n) 
		y_tridiag.append(time.time()-start_time2)


		# start profiling of our LU
		start_time = time.time()  
		res1= lu_1(A) 
		y_notre.append(time.time()-start_time)


		# start profiling LU of scipy 
		start_time2 = time.time() 
		res2= lu_factor(A) 
		y_scipy.append(time.time()-start_time) 

		# incrémenter la taille de la matrice
		n=n+5 

	return y_scipy, y_notre, y_tridiag, x ,A,res2 





#----------------------------------------------------------------------------------------------------------------------------------------------

def tridiag(a, b, c): 
	# construction del a matrice tridiagonale 
	return np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)


#----------------------------------------------------------------------------------------------------------------------------------------------


def lu_1(A):  
	# récupérer la ttaille de la matrice  A 
    n=len(A) 
	# initialiser L 
    L = np.eye(len(A)) 
	# initialiser U 
    U = np.copy(A)
     	
	# parcourir la matrice A ligne par ligne 
    for i in range(n):
	pivot = U[i,i] 
	assert pivot != 0, "Non respect de condition nécéssaire et suffisante de factorisation LU"
	for j in range(i+1,n): 
		# diviser les éléments de la diagonale suppérieure de U par le pivot 
		s= -U[j,i]/pivot   
		
		# mettre à jour la ligne de la matrice U par combinaison linéaire 
		U[j] = U[j]+(s*U[i])

		# mettre à jour les éléments L[j,i]
		L[j,i] = -s 
    L=np.eye(n)+ L  
    return L,U
    





#---------------------------------------------------------------------------------------------------------------




def lu_tridiag(A,l,u,v,a,n):  

	# LU sur matrice tridiagonale a une complexité linéaire , il suffit d'une seule boucle 
	for i in range(1,n): 
		l[i]= a[i-1]/v[i-1] 
		v[i]=A[i,i]-(l[i]*u[i-1])



	l= np.delete(l, 0) 

	# construire L ET U à partir de u,v,a,l
	L= np.diag(l,-1) + np.eye(n,n)
	U= np.diag(v,0)+ np.diag(u,1)
	return L,U 






#-------------------------------------------------------------------------------------------------------------------------------------

def solve(A,n,N):  


	s= [ -2 for i in range(n)]  

	# définir le nombre de pas = 4 donc N+1  = 5   

	# définir le pas H 
	h= 1/(N+1) 


	# initialiser le vecteur S aux extrémités 
	s[0] = 0/(h*h)
	s[n-1]= 1/(h*h) 


	#calculer le déterminant de la matrice A 
	det_A= np.linalg.det(A) 
#  si la matrice est inversible donc det(A) != 0 
	if (det_A != 0):   

		# inverser la matrice A 
		inv_A= np.linalg.inv(A)   

		# résolution du système 
		u = np.dot(h*h*inv_A,s)  
		sol= -(u) 



	return sol,h




#----------------------------------------------------------------------------------------------------------------------


def plot_result(h,sol): 
	axe_x = []
	axe_y= []
	axe_y1=[]

	h= 1/(N+1) 



	# remplire le vecteur de axes  x et y 
	for i in range(5): 
		axe_x.append(i*h)   
		axe_y1.append((i*h)*(i*h))
		axe_y.append(sol[i])
		print(i*h)

	axe_x.append(1)
	axe_y.append(1)
	axe_y1.append(1)


	t= np.arange(0,1,0.1)
	print("vecteur t -----------",t)
	plt.plot(axe_x,axe_y,c="b")
	plt.show()







#programme principal 
#-------------------------------------------------------------------------------------------------------------------------------------



# initialiser la taille des données 
n=5 

# initialiser la taille des sous intervales [0,1]
N= 4 

a = [1 for i in range(n-1)]; b = [-2 for i in range(n)]; c = [1 for i in range(n-1)] 
# initialiser la matrice tridiagonale A 
A = tridiag(a, b, c) 

res= test_lu(n) 





# profiling de résolution lu sans scipy sans lu 
start_time3 = time.time()
result= solve(A,n,N)
print("temps d'exécution résolution sans scipy, sans lu",time.time()-start_time3)




#----------------------- manque la résolution à la main avec LU, méthode ascendante, descendate



# profiling de la résolution avec lu + scipy 
lu, piv = lu_factor(A)
start_time4 = time.time()
x = lu_solve((lu, piv), result[0])
print("temps de résolution avec scipy lu",time.time()-start_time4)  






plot_result(result[1],result[0])
# tracer le graphe de performance de la factorusation LU 
plt.plot(res[3], res[0], label="scipy")
plt.plot(res[3], res[1], label="notre")
plt.plot(res[3], res[2], label="tridiag")



#------------- manque plot_result_resolution() pour analyser les différents temps d'exécution de la résolution 

plt.legend()
plt.show() 














