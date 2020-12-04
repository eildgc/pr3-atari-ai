# pr3-atari-ai
Proyecto #3 y entrega final de la materia Inteligencia Artificial. Desarrollo de un agente de Inteligencia Artificial que sea capaz de jugar un juego de Atari utilizando OpenAI Gym como entorno.

## Aprendizaje forzado e implementación
Inicialmente en el aprendizaje forzado (a diferencia del aprendizaje supervisado) se desconoce qué está bien o qué está mal de nuestra información, la única forma de saber esto al inicio es probando y ver si funciona. Se auto generan variables a traves de observaciones y retroalimentación.

Uno de los ejemplos que utiliza Soroush (thinkingparticle) para comprender mejor lo que es el aprendizaje forzado y sus implicaciones, nos pide que nos imaginemos un mundo imaginario donde somos gerentes de un nuevo departamento y no conocemos a nuestros empleados, no tenemos acceso a sus curriculum ni registros. No sabemos cual empleado es un diseñador, un programador, o del área de atención al cliente, etc.

Además de esto consideramos que no podemos comunicarnos con los empleados, solo podemos asignarles tareas a través del panel de una computadora y ver que tan rápido finalizan su asignación o verlos fallar.

El acercamiento a la solución de esta situación que propone es intentar y fallar, registrando la experiencia resultante. Entonces debemos ver los resultados y suponer los roles de los empleados. Y si actuamos conforme a esto y acertamos en nuestra supocision, entones repetimos el proceso hasta que conozcamos los roles de todos.

Agente: es aquel que está aprendiendo a ganar y adquiere beneficios para sí mismo (recompensas)
Ambiente(environment): El mundo con el que el agente puede interactuar. El agente "ve" states y hace "acciones" de acuerdo a lo que ve.
Recompensa(reward): El agente obtiene recompensas o castigos según sus acciones. 

## 

A systematic study on the Pong environment by Phon-Amnuaisuk suggested that roughly 10000 episodes of data is needed to generate an agent capable of achieving a win.
https://arxiv.org/pdf/1807.08452.pdf
## Fuentes y referencias
https://gym.openai.com/envs/#atari
https://github.com/thinkingparticle/deep_rl_pong_keras
https://nbviewer.jupyter.org/github/thinkingparticle/deep_rl_pong_keras/blob/master/reinforcement_learning_pong_keras_policy_gradients.ipynb