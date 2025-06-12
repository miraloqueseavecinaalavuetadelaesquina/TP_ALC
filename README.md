## CORRECIONES TP1
Corrector: Ariel

Desgloce por ejercicio (R revisar, B bien)

BRRBRR

Comentarios

- En general: dejen los resultados de las celdas de código ya corridas, de forma tal de poder revisar más fácilmente los resultados.

- En general: Su código no corre y hay celdas que generan funciones que no funcionan (tienen aun los ...). Asegurense antes de entregar que todo corra.

- Punto 2: Las funciones deben tener comentarios indicando que hacen.

- Punto 2: La demostración es errónea, ya que lo que se pide es mostrar que M no es singular, no A ni C. Revisen. Sin haber podido probar esto, ¿Aqué garantiza que todos los pasos que se realizan de aquí en adelante tengan sentido siempre?

- Punto 6: Falta

## FALTA CORREGIR
- EJERCICIO 2: Puse el punto 2 de Migue.
- EJERCICIO 3: Falta el punto 3.d.b final (en la primer entrega no estaba ese punto así que no lo tengo jajaja). Para el punto 3.d.a agregué 2 funciones para calcular el historial de top 3 museos según $m$ y $\alpha$ según corresponda. Las funciones son `analiza_top_museos` y `grafica_evolucion` (Migue).
- EJERCICIO 5: Agregado (Anto). Sobre lo que hicieron agregue modificaciones sobre las funciones `calcula_B()` (modifique las cotas del ciclo, la cual iniciaba en 0 y terminaba en 1-cantidad de visitas, lo cual generaba error en el cálculo de B durante las primeras iteraciones) (Migue).
- EJERCICIO 6: Agregado (Migue).

## ORGANIAZACION DEL REPO
- RECURSOS TP1 Y TP2: los archivos que nos mandaron para hacer los tps
- ENTREGA: lo que entregamos para el tp1. 
La idea seria ir corrigiendo los archivos funciones.py y TP.ipynb que es lo que vamos a terminar entregando.

## COMENTARIOS REENTREGA TP1
Habría que organizar mejor las funciones utilizadas. Actualmente como dejé el notebook unicamente utiliza funciones definidas dentro del mismo notebook. Sería mejor guardar estas funciones en un archivo aparte del tipo `funciones.py` e importarlo para que sea mas legible el informe.
Por último, sientanse libres de testear cualquier función agregada o modificación realizada :D