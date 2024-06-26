import matplotlib.pyplot as plt

##RMSE
modeles = ['Régression linéaire multiple', 'KNN', 'Random Forest']
RMSE = [0.08, 0.1, 0.063]

plt.figure(figsize=(10, 6))
plt.bar(modeles, RMSE, color='skyblue')

plt.title('Performance des modèles en termes de RMSE')
plt.xlabel('Modèles')
plt.ylabel('RMSE')
plt.xticks(rotation=45, ha='right') 
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#MEA
modeles = ['Régression linéaire multiple', 'KNN', 'Random Forest']
MEA = [0.06,0.07, 0.043]

plt.figure(figsize=(10, 6))
plt.bar(modeles, MEA, color='lightgreen')

plt.title('Performance des modèles en termes de MAE')
plt.xlabel('Modèles')
plt.ylabel('MAE')
plt.xticks(rotation=45, ha='right')  
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#MSE

modeles = ['Régression linéaire multiple', 'KNN', 'Random Forest']
MSE = [0.01, 0.01, 0.004]

plt.figure(figsize=(10, 6))
plt.bar(modeles, MSE, color='orange')

plt.title('Performance des modèles en termes de MSE')
plt.xlabel('Modèles')
plt.ylabel('MSE')
plt.xticks(rotation=45, ha='right')  # Rotation des étiquettes sur l'axe x pour une meilleure lisibilité
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()