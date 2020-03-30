# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:08:43 2020

"""
#import packages
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import numpy as np
import networkx as nx
def modelEvaluation(real_matrix,predict_matrix,testPosition,featurename): #  evaluation
       real_labels=[]
       predicted_probability=[]

       for i in range(0,len(testPosition)):
           real_labels.append(real_matrix[testPosition[i][0],testPosition[i][1]])
           predicted_probability.append(predict_matrix[testPosition[i][0],testPosition[i][1]])

#       predicted_probability= normalize.fit_transform(predicted_probability)
       real_labels=np.array(real_labels)
       predicted_probability=np.array(predicted_probability)
       predicted_probability=predicted_probability.reshape(-1,1)
       precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
       aupr_score = auc(recall, precision)

       all_F_measure=np.zeros(len(pr_thresholds))
       for k in range(0,len(pr_thresholds)):
           if (precision[k]+precision[k])>0:
              all_F_measure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
           else:
              all_F_measure[k]=0
       max_index=all_F_measure.argmax()
       threshold=pr_thresholds[max_index]

       fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
       auc_score = auc(fpr, tpr)
       predicted_score=np.zeros(len(real_labels))
       predicted_score=np.where(predicted_probability > threshold, 1, 0)

       f=f1_score(real_labels,predicted_score)
       accuracy=accuracy_score(real_labels,predicted_score)
       precision=precision_score(real_labels,predicted_score)
       recall=recall_score(real_labels,predicted_score)
       print('results for feature:'+featurename)
       print('************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f, f-measure:%.3f************************' %(auc_score,aupr_score,recall,precision,accuracy,f))
       auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), ("%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)
       results=[auc_score,aupr_score,precision, recall,accuracy,f]
       return results

#-------------------------------------------------
def A_matrix(S_c,S_d, q):
    A_temp=np.zeros([q,q])
    A_hat=np.zeros([S_c.shape[0],S_d.shape[0]])

    for i in range(0,q):
        drug_index=int(i/S_d.shape[0])
        disease_index=int(i%S_d.shape[0])

        A_temp[i,i]=np.sum(S_c[drug_index])*np.sum(S_d[disease_index])-1
        A_hat[drug_index][disease_index]=np.sqrt(A_temp[i,i])
    return A_temp,A_hat
def create_P(S_G,drug_gene_interaction_C,disease_gene_interaction_D,Y):
    P_temp=np.zeros(Y.shape[0],Y.shape[1])
    for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if Y[i,j]==0:
                    P_temp[i,j]=np.multiply(np.multiply(drug_gene_interaction_C[i].T,S_G),disease_gene_interaction_D[j])/(np.sqrt(np.multiply(np.multiply(drug_gene_interaction_C[i].T,S_G),drug_gene_interaction_C[i]))*np.sqrt(np.multiply(np.multiply(disease_gene_interaction_D[j].T,S_G),disease_gene_interaction_D[j])))
    return P_temp

#-----------------------------------------------------------
def create_GSim(gene_gene_interaction,num_g):
    S_G=np.zeros(num_g,num_g)
    G=nx.read_edgelist(gene_gene_interaction)
    path=nx.all_pairs_shortest_path(G)
    for i in range(num_g):
        for j in range(num_g):
            D=len(path[i,j])
            S_G[i,j]=0.3*np.exp(-0.1*D)
            
    #shoretest path
    return S_G
#-----------------------------------------------------------------------------
def sim_treatment(Y,inputType):
    G=nx.Graph()
    G.add_node(np.arange(Y.shape[0]),bipartite=0)
    G.add_node(np.arange(Y.shape[0],Y.shape[0]+Y.shape[1]),bipartite=1)   
    for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if Y[i,j]==1:
                    G.add_edge(i,Y.shape[0]+j)
    path=nx.all_pairs_shortest_path(G)
    if inputType=='drug':
     S_ct=np.zeros(Y.shape[0],Y.shape[0])

     for i in range(Y.shape[0]):
        for j in range(Y.shape[0]):
            D=len(path[i,j])
            S_ct[i,j]=0.3*np.exp(-0.1*D)
     return S_ct
    if inputType=='disease':
     S_dt=np.zeros(Y.shape[1],Y.shape[1])

     for i in range(Y.shape[1]):
        for j in range(Y.shape[1]):
            D=len(path[i+Y.shape[0],j+Y.shape[0]])
            S_dt[i,j]=0.3*np.exp(-0.1*D)
     return S_dt
#----------------------------------------------------------------           
def create_MySim(sim,Y,S_G,inter,alpha,gamma,beta):
    sim_G=np.zeros(sim.shape[0],sim.shape[0])
    for i  in range(sim.shape[0]):
        for j in range(sim.shape[0]):
            sim_G[i,j]=np.multiply(np.multiply(inter[i].T,S_G),inter[j])/(np.sqrt(np.multiply(np.multiply(inter[i].T,S_G),inter[i]))*np.sqrt(np.multiply(np.multiply(inter[j].T,S_G),inter[j])))
    S_t= sim_treatment(Y)
    
    Sim_total=alpha*sim+beta*sim_G+gamma*S_t
    return Sim_total
#----------------------------------------------------------------------
def SSGC(S_c,S_d,Y,miu,P,zeta,tol):
    m=Y.shape[0]
    n=Y.shape[1]
    alpha=1/(1+miu)
    U_temp=np.ones([m,n])
    U=U_temp-Y
    Y_hat=Y+((zeta/miu)*P)
    A_m,A_hat=A_matrix(S_c,S_d,n*m)
    A_tilda=np.multiply(A_hat,A_hat)
    F_old=Y_hat
    F_new=Y_hat
    convergence=False
    
    while not convergence:
        part1=(miu-zeta)*np.multiply(U,F_old)
        part2=np.dot(np.dot(S_c,(np.divide(F_old,A_hat))),S_d)
        F_new=alpha*(part1+np.divide(part2,A_hat)-np.divide(F_old,A_tilda))+(1-alpha)*Y_hat
        
        if np.sum(F_new-F_old)<tol:
            convergence=True
            return F_new
        F_old=F_new

#----------------------------------------------------------------
    
def main():
   # inputs from user
   #Drug diseas assosiation Matrix
#   Y=np.loadtxt('mat_drug_disease.txt',delimiter=' ')
#   S_c=np.loadtxt('Sim_mat_drug_protein.txt',delimiter='\t')
#   S_d=np.loadtxt('Sim_mat_disease_protein.txt',delimiter='\t')
   Y= np.random.choice([0, 1], size=(100,200), p=[2./3, 1./3])
   S_c=np.random.rand(100,100)
   S_d=np.random.rand(200,200)
   #hyperparameters from input
   #-------------------------
   
   alpha=0.7
   beta=1
   gamma=1
   miu=4
   zeta=0.67
   # tolerance for convergence
   tol=10
   #----------------------------------
   #uncomment if you have all of information
   
#   
#   #Drug Substructure Similarity that calculated from fingerprint of drugs
#   drug_chemical_sim=np.loadtxt('')
#   # Drug Phenotype similarity Matrix calculated using Mesh
#   diseasPhenotype_sim=np.loadtxt('')
#   # drug gene interaction Matrix from DrugBank
#   drug_gene_interaction_C=np.loadtxt('')
#   #disease gene interaction Matrix from Mesh 
#   disease_gene_interaction_D=np.loadtxt('')
#   #Gene Gene interaction in two column format 
#   gene_gene_interaction=np.loadtxt('')
#   #calculate number of genes
#   num_g=drug_gene_interaction_C.shape[1]
#   #calculate similarity beween each genes using gene gene interaction matrix by create GSim function
#   S_G=create_GSim(gene_gene_interaction,num_g)
#   #calculate final similarity matrix for diseases an drugs by create_Mysim function. this function needs apha, beta, gamma and interaction matrix and another sim matrix
#   S_c=create_MySim(drug_chemical_sim,Y,S_G,drug_gene_interaction_C,alpha,gamma,beta)
#   S_d=create_MySim(diseasPhenotype_sim,Y,S_G,disease_gene_interaction_D,alpha,gamma,beta)
#   #calculate prior knowledge using Similarity of genes that related to drugs and diseases.
#   P=create_P(S_G,drug_gene_interaction_C,disease_gene_interaction_D,Y)
#   # call SSGC Algorithm for calculate predicted matrix

   P=np.random.rand(Y.shape[0],Y.shape[1])
   seed=0
   link_number = 0
   CV_num=10
   link_position = []
   nonLinksPosition = []  # all non-link position
   for i in range(0, len(Y)):
        for j in range(0, len(Y[0,])):
            if Y[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            else:
                nonLinksPosition.append([i, j])

   link_position = np.array(link_position)
   random.seed(seed)
   index = np.arange(0, link_number)
   random.shuffle(index)
   fold_num = link_number//CV_num
   print(fold_num)
   for CV in range(0, CV_num):
        print('*********round:' + str(CV) + "**********\n")
        test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
        test_index.sort()
        testLinkPosition = link_position[test_index]
        train_drug_des_matrix = copy.deepcopy(Y)
        for i in range(0, len(testLinkPosition)):
            train_drug_des_matrix[testLinkPosition[i, 0], testLinkPosition[i, 1]] = 0
#            train_drug_des_matrix[testLinkPosition[i, 1], testLinkPosition[i, 0]] = 0
            testPosition = list(testLinkPosition) + list(nonLinksPosition)
        Predicted_matrix=SSGC(S_c,S_d,train_drug_des_matrix,miu,P,zeta,tol)
        results =modelEvaluation(Y,Predicted_matrix,testPosition,'DrugDisInt')
        print(results)
        np.savetxt('Predictedmat.txt',Predicted_matrix)
#call main function
main()
