import os 
import time  
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
import random 
import copy  
from dataclasses import dataclass, field  
from typing import List, Tuple, Dict  

@dataclass
class HojaMaterial:
    ancho: float
    alto: float  

@dataclass
class TipoElemento:
    id: int  
    ancho: float
    alto: float
    cantidad: int  
    nombre: str  

@dataclass
class Elemento:
    id_tipo: int
    ancho: float
    alto: float
    nombre: str

@dataclass
class TPila:
    elementos: List[Elemento] = field(default_factory=list)
    ancho: float = 0.0  
    alto_usado: float = 0.0  

@dataclass
class TTira:
    pilas: List[TPila] = field(default_factory=list)  
    alto: float = 0.0 
    ancho_usado: float = 0.0 

@dataclass
class THojaCortada:
    tiras: List[TTira] = field(default_factory=list)  
    alto_usado: float = 0.0  
TCortado = List[THojaCortada]

@dataclass
class Individuo:

    genes: List[int]
    cortado: TCortado = field(default_factory=list)
    aptitud: float = float('inf')  

    def __lt__(self, other):
        return self.aptitud < other.aptitud

# ---------- CLASE PRINCIPAL DEL ALGORITMO EVOLUTIVO ----------

class AlgoritmoEvolutivo:
    
    def __init__(self, tam_poblacion: int, prob_cruce: float, prob_mutacion: float, num_generaciones: int):
        self.tam_poblacion = tam_poblacion  # Cuántos individuos (soluciones) probar en cada generación
        self.prob_cruce = prob_cruce  # Probabilidad de que dos padres se "reproduzcan"
        self.prob_mutacion = prob_mutacion  # Probabilidad de que un individuo "mute" (cambie aleatoriamente)
        self.num_generaciones = num_generaciones  # Cuántas veces repetir el ciclo de evolución
        
        # --- Variables de estado ---
        self.hoja_base: HojaMaterial = None 
        self.tipos_elementos: List[TipoElemento] = []  
        self.poblacion: List[Individuo] = [] 
        self.mapa_tipos: Dict[int, TipoElemento] = {}  

    def cargar_piezas(self, hoja: HojaMaterial, piezas_a_cargar: List[Dict]):

        self.hoja_base = hoja
        self.tipos_elementos = []  
        self.mapa_tipos = {} 
        
        id_counter = 0
        for pieza_dict in piezas_a_cargar:
            ancho = pieza_dict['ancho']
            alto = pieza_dict['alto']
            cantidad = pieza_dict['cantidad']
            
            tipo = TipoElemento(
                id=id_counter,
                ancho=ancho,
                alto=alto,
                cantidad=cantidad,
                nombre=f"Pieza {id_counter+1} ({ancho}x{alto})"
            )
            self.tipos_elementos.append(tipo)  
            self.mapa_tipos[id_counter] = tipo  
            id_counter += 1

        print(f"Plano base: {self.hoja_base.ancho} x {self.hoja_base.alto}")
        print(f"Cargados {len(self.tipos_elementos)} tipos de piezas únicas.")

    def generar_poblacion_inicial(self):

        print("Generando población inicial...")
        indices_tipos = [tipo.id for tipo in self.tipos_elementos]
        
        for _ in range(self.tam_poblacion):
            genes_permutados = random.sample(indices_tipos, len(indices_tipos))
            # Crea un nuevo Individuo con esos genes y lo agrega a la población
            self.poblacion.append(Individuo(genes=genes_permutados))
    
    def calcular_aptitud(self, individuo: Individuo):

        elementos_a_colocar: List[Elemento] = []
        for id_tipo in individuo.genes: 
            tipo = self.mapa_tipos[id_tipo]
            for _ in range(tipo.cantidad):
                elementos_a_colocar.append(Elemento(
                    id_tipo=tipo.id,
                    ancho=tipo.ancho,
                    alto=tipo.alto,
                    nombre=tipo.nombre
                ))

        patron_cortado: TCortado = [THojaCortada()]  
        hoja_actual = patron_cortado[-1]
        
        tira_actual = TTira()
        pila_actual = TPila()
        
        W_HOJA = self.hoja_base.ancho  
        L_HOJA = self.hoja_base.alto 

        for i, elem in enumerate(elementos_a_colocar):
            colocado = False  

        for i, elem in enumerate(elementos_a_colocar):
            colocado = False
            
            orientaciones = [(elem.ancho, elem.alto, False)]
            if elem.ancho != elem.alto:
                orientaciones.append((elem.alto, elem.ancho, True))

            if not colocado and hoja_actual.tiras:
                tira_actual = hoja_actual.tiras[-1] 
                
                for ancho_p, alto_p, es_rotada in orientaciones:
                    # Chequeo 1: ¿Cabe en la última PILA de la tira? (Mismo ancho)
                    if tira_actual.pilas:
                        pila_actual = tira_actual.pilas[-1]
                        if (ancho_p == pila_actual.ancho and 
                            alto_p + pila_actual.alto_usado <= tira_actual.alto):
                            
                            nombre_final = elem.nombre + " (R)" if es_rotada else elem.nombre
                            nuevo_elem = Elemento(elem.id_tipo, ancho_p, alto_p, nombre_final)
                            pila_actual.elementos.append(nuevo_elem)
                            pila_actual.alto_usado += alto_p
                            colocado = True
                            break 
                    
                    # Chequeo 2: ¿Cabe como NUEVA PILA en esta misma tira?
                    if not colocado:
                        if (ancho_p + tira_actual.ancho_usado <= W_HOJA and
                            alto_p <= tira_actual.alto):
                            
                            nombre_final = elem.nombre + " (R)" if es_rotada else elem.nombre
                            nuevo_elem = Elemento(elem.id_tipo, ancho_p, alto_p, nombre_final)
                            nueva_pila = TPila(ancho=ancho_p, alto_usado=alto_p)
                            nueva_pila.elementos.append(nuevo_elem)
                            
                            tira_actual.pilas.append(nueva_pila)
                            tira_actual.ancho_usado += ancho_p
                            colocado = True
                            break

            # --- NIVEL 2: ABRIR NUEVA TIRA EN LA HOJA ACTUAL ---
            if not colocado:
                # Probamos rotaciones para ver cuál consume menos alto o si alguna entra
                for ancho_p, alto_p, es_rotada in orientaciones:
                    if (ancho_p <= W_HOJA and 
                        alto_p + hoja_actual.alto_usado <= L_HOJA):
                        
                        nombre_final = elem.nombre + " (R)" if es_rotada else elem.nombre
                        nuevo_elem = Elemento(elem.id_tipo, ancho_p, alto_p, nombre_final)
                        
                        nueva_tira = TTira(alto=alto_p, ancho_usado=ancho_p)
                        nueva_pila = TPila(ancho=ancho_p, alto_usado=alto_p)
                        nueva_pila.elementos.append(nuevo_elem)
                        nueva_tira.pilas.append(nueva_pila)
                        
                        hoja_actual.tiras.append(nueva_tira)
                        hoja_actual.alto_usado += alto_p
                        colocado = True
                        break

            # --- NIVEL 3: ABRIR NUEVA HOJA ---
            if not colocado:
                # Si llegamos aquí, la pieza no cabe en la hoja actual de ninguna forma
                ancho_p, alto_p = elem.ancho, elem.alto
                nombre_final = elem.nombre
                
                if ancho_p > W_HOJA or alto_p > L_HOJA:
                     if alto_p <= W_HOJA and ancho_p <= L_HOJA:
                         ancho_p, alto_p = alto_p, ancho_p
                         nombre_final += " (R)"
                     else:
                        print(f"ERROR: {elem.nombre} es demasiado grande para el plano.")
                        continue

                hoja_actual = THojaCortada()
                patron_cortado.append(hoja_actual)
                
                nueva_tira = TTira(alto=alto_p, ancho_usado=ancho_p)
                nueva_pila = TPila(ancho=ancho_p, alto_usado=alto_p)
                nuevo_elem = Elemento(elem.id_tipo, ancho_p, alto_p, nombre_final)
                
                nueva_pila.elementos.append(nuevo_elem)
                nueva_tira.pilas.append(nueva_pila)
                hoja_actual.tiras.append(nueva_tira)
                hoja_actual.alto_usado = alto_p

                
                
                # ---- INICIO LÓGICA DE 3 ETAPAS ----
                
                # Etapa 0: ¿La hoja está vacía? (Es la primera pieza de la hoja)
                if not hoja_actual.tiras:
                    if ancho > W_HOJA or alto > L_HOJA:
                        continue 
                    tira_actual = TTira(alto=alto, ancho_usado=ancho)
                    pila_actual = TPila(ancho=ancho, alto_usado=alto)
                    pila_actual.elementos.append(elem_rotado)
                    tira_actual.pilas.append(pila_actual)
                    hoja_actual.tiras.append(tira_actual)
                    hoja_actual.alto_usado = alto
                    colocado = True
                    break  

                # Etapa 3: ¿Cabe en la PILA actual?
                if (ancho == pila_actual.ancho and 
                    alto + pila_actual.alto_usado <= tira_actual.alto):
                    
                    pila_actual.elementos.append(elem_rotado)
                    pila_actual.alto_usado += alto
                    colocado = True
                    break

                # Etapa 2: ¿Cabe en la TIRA actual (como nueva PILA)?
                if (ancho + tira_actual.ancho_usado <= W_HOJA and
                    alto <= tira_actual.alto):
                    
                    pila_actual = TPila(ancho=ancho, alto_usado=alto)
                    pila_actual.elementos.append(elem_rotado)
                    tira_actual.pilas.append(pila_actual)
                    tira_actual.ancho_usado += ancho
                    colocado = True
                    break

                # Etapa 1: ¿Cabe en la HOJA actual (como nueva TIRA)?
                if (ancho <= W_HOJA and
                    alto + hoja_actual.alto_usado <= L_HOJA):
                    
                    tira_actual = TTira(alto=alto, ancho_usado=ancho)
                    pila_actual = TPila(ancho=ancho, alto_usado=alto)
                    pila_actual.elementos.append(elem_rotado)
                    tira_actual.pilas.append(pila_actual)
                    hoja_actual.tiras.append(tira_actual)
                    hoja_actual.alto_usado += alto
                    colocado = True
                    break

            # Etapa 0: No cupo en ningún lado de la hoja actual
            if not colocado:
                # Volvemos a la pieza sin rotar (por si la rotación falló)
                ancho, alto, nombre = elem.ancho, elem.alto, elem.nombre
                elem_rotado = elem
                
                if ancho > W_HOJA or alto > L_HOJA:
                    print(f"ERROR: Elemento {elem.nombre} es más grande que la hoja.")
                    continue  

                hoja_actual = THojaCortada()
                patron_cortado.append(hoja_actual)
                
                tira_actual = TTira(alto=alto, ancho_usado=ancho)
                pila_actual = TPila(ancho=ancho, alto_usado=alto)
                pila_actual.elementos.append(elem_rotado)
                tira_actual.pilas.append(pila_actual)
                hoja_actual.tiras.append(tira_actual)
                hoja_actual.alto_usado = alto
        
        individuo.cortado = patron_cortado  # Guardamos el fenotipo en el individuo

        S_x = len(patron_cortado)  
        if S_x == 0:
            individuo.aptitud = 0
            return

        c_L = patron_cortado[-1].alto_usado
        L = self.hoja_base.alto  

        if L > 0:
             individuo.aptitud = S_x - ((L - c_L) / L)
        else:
             individuo.aptitud = S_x  

    def seleccionar_padres(self) -> Tuple[Individuo, Individuo]:

        k_torneo = 3
        if len(self.poblacion) < k_torneo:
            k_torneo = len(self.poblacion)
            
        padre1 = min(random.sample(self.poblacion, k=k_torneo), key=lambda i: i.aptitud)
        padre2 = min(random.sample(self.poblacion, k=k_torneo), key=lambda i: i.aptitud)
        return padre1, padre2

    def cruce_por_orden(self, padre1: Individuo, padre2: Individuo) -> Tuple[Individuo, Individuo]:
        genes1 = padre1.genes
        genes2 = padre2.genes
        n = len(genes1)
        
        hijo1_genes = [None] * n
        hijo2_genes = [None] * n
        
        if n == 0:
             return Individuo(genes=[]), Individuo(genes=[])
        if n == 1:
             return Individuo(genes=genes1), Individuo(genes=genes2)

        pto1, pto2 = sorted(random.sample(range(n), 2))
        
        hijo1_genes[pto1:pto2+1] = genes1[pto1:pto2+1]
        hijo2_genes[pto1:pto2+1] = genes2[pto1:pto2+1]
        
        
        # Rellenar Hijo 1 con genes de Padre 2
        genes_padre2 = genes2[pto2+1:] + genes2[:pto2+1] 
        idx_hijo1 = pto2 + 1
        for gen in genes_padre2:
            if gen not in hijo1_genes:
                hijo1_genes[idx_hijo1 % n] = gen 
                idx_hijo1 += 1
                
        # Rellenar Hijo 2 con genes de Padre 1
        genes_padre1 = genes1[pto2+1:] + genes1[:pto2+1]
        idx_hijo2 = pto2 + 1
        for gen in genes_padre1:
            if gen not in hijo2_genes:
                hijo2_genes[idx_hijo2 % n] = gen
                idx_hijo2 += 1
        
        return Individuo(genes=hijo1_genes), Individuo(genes=hijo2_genes)

    def mutacion(self, individuo: Individuo):

        genes = individuo.genes
        n = len(genes)
        if n < 2: return 

        if random.random() < 0.5:
            idx1, idx2 = random.sample(range(n), 2)
            genes[idx1], genes[idx2] = genes[idx2], genes[idx1]
        else:
            max_len_bloque = n // 2
            if max_len_bloque < 1: return
            
            longitud = random.randint(1, max_len_bloque)
            idx1 = random.randint(0, n - 1 - longitud)
            idx2 = random.randint(0, n - 1 - longitud)
            
            intentos = 0
            while max(idx1, idx2) < min(idx1, idx2) + longitud and intentos < 10:
                idx2 = random.randint(0, n - 1 - longitud)
                intentos += 1
            
            if intentos >= 10: return  

            bloque1 = genes[idx1 : idx1 + longitud]
            bloque2 = genes[idx2 : idx2 + longitud]
            genes[idx1 : idx1 + longitud] = bloque2
            genes[idx2 : idx2 + longitud] = bloque1
            
        individuo.genes = genes  

    def ejecutar(self) -> Individuo:

        if not self.hoja_base or not self.tipos_elementos:
            print("Error: Debes cargar los datos primero.")
            return None

        # 1. Crear la población inicial aleatoria
        self.generar_poblacion_inicial()
        
        if not self.poblacion:
            print("Error: No se pudo generar la población inicial (¿no hay piezas?).")
            return None

        # 2. Evaluar la población inicial (Generación 0)
        print("Evaluando población inicial (Generación 0)...")
        for ind in self.poblacion:
            self.calcular_aptitud(ind)  # Calcular la puntuación de cada individuo
        
        mejor_individuo = min(self.poblacion)  # Encontrar al mejor de la Gen 0
        print(f"Generación 0: Mejor Aptitud = {mejor_individuo.aptitud:.4f}")

        # 3. Iniciar el Ciclo Evolutivo
        for gen in range(1, self.num_generaciones + 1):
            nueva_poblacion = []
            nueva_poblacion.append(copy.deepcopy(mejor_individuo))
            
            # Llenar el resto de la nueva población
            while len(nueva_poblacion) < self.tam_poblacion:
                # 4. Selección
                padre1, padre2 = self.seleccionar_padres()
                
                # 5. Cruce (Reproducción)
                if random.random() < self.prob_cruce:
                    hijo1, hijo2 = self.cruce_por_orden(padre1, padre2)
                else:
                    # Si no hay cruce, los hijos son clones de los padres
                    hijo1, hijo2 = copy.deepcopy(padre1), copy.deepcopy(padre2)
                
                # 6. Mutación
                if random.random() < self.prob_mutacion:
                    self.mutacion(hijo1)
                if random.random() < self.prob_mutacion:
                    self.mutacion(hijo2)
                
                # 7. Evaluación de los nuevos hijos
                self.calcular_aptitud(hijo1)
                self.calcular_aptitud(hijo2)
                
                # 8. Agregar hijos a la nueva población
                nueva_poblacion.extend([hijo1, hijo2])

            # La nueva generación reemplaza a la antigua
            self.poblacion = nueva_poblacion[:self.tam_poblacion]
            
            # Encontrar al mejor de la generación actual
            mejor_individuo_gen = min(self.poblacion)
            
            # Actualizar el mejor de todos los tiempos
            if mejor_individuo_gen.aptitud < mejor_individuo.aptitud:
                mejor_individuo = copy.deepcopy(mejor_individuo_gen)
            
            if gen % 10 == 0 or gen == self.num_generaciones:
                print(f"Generación {gen}: Mejor Aptitud = {mejor_individuo.aptitud:.4f}")

        print("\n--- Evolución Terminada ---")
        print(f"Mejor individuo encontrado (Aptitud = {mejor_individuo.aptitud:.4f}):")
        orden_tipos = [self.mapa_tipos[id_gen].nombre for id_gen in mejor_individuo.genes]
        print(f"Orden de tipos (Genotipo): {orden_tipos}")
        return mejor_individuo  



# ---------- VISUALIZACIÓN (Matplotlib) ----------

def dibujar_patron_de_corte(hoja_base: HojaMaterial, patron: TCortado):
    
    num_hojas = len(patron)
    fig, axs = plt.subplots(1, num_hojas, figsize=(7 * num_hojas, 10))
    if num_hojas == 1:
        axs = [axs]  

    for i, hoja_cortada in enumerate(patron):
        ax = axs[i]
        ax.set_title(f"Algoritmo Genético - Plano {i + 1}")
        ax.set_xlim(0, hoja_base.ancho)
        ax.set_ylim(0, hoja_base.alto)
        ax.set_xlabel("Ancho (cm)")
        ax.set_ylabel("Alto (Largo) (cm)")
        
        ax.add_patch(patches.Rectangle((0, 0), hoja_base.ancho, hoja_base.alto, 
                                     linewidth=2, edgecolor='black', facecolor='whitesmoke', zorder=0))
        
        y_cursor_tira = 0  
        
        # Itera sobre las Tiras (Etapa 1)
        for tira in hoja_cortada.tiras:
            ax.add_patch(patches.Rectangle((0, y_cursor_tira), hoja_base.ancho, tira.alto, 
                                         linewidth=1.5, edgecolor='blue', facecolor='none', 
                                         linestyle='--', zorder=1))
            
            x_cursor_pila = 0  
            # Itera sobre las Pilas (Etapa 2)
            for pila in tira.pilas:
                color_pila = (random.random(), random.random(), random.random()) 
                ax.add_patch(patches.Rectangle((x_cursor_pila, y_cursor_tira), pila.ancho, tira.alto, 
                                             linewidth=1, edgecolor='red', facecolor='none', 
                                             linestyle=':', zorder=2))
                
                y_cursor_elem = y_cursor_tira  
                
                # Itera sobre los Elementos (Etapa 3)
                for elem in pila.elementos:
                    rect = patches.Rectangle((x_cursor_pila, y_cursor_elem), elem.ancho, elem.alto,
                                             linewidth=1, edgecolor='black', facecolor=color_pila, 
                                             alpha=0.8, zorder=3)
                    ax.add_patch(rect)
                    
                    texto = f"{elem.nombre}\n{elem.ancho}x{elem.alto}"
                    if elem.ancho > 4 and elem.alto > 4:  
                         ax.text(x_cursor_pila + elem.ancho / 2, y_cursor_elem + elem.alto / 2, 
                                 texto, ha="center", va="center", fontsize=6, color="black")
                    
                    y_cursor_elem += elem.alto  
                
                x_cursor_pila += pila.ancho 
            
            y_cursor_tira += tira.alto  
            
        ax.set_aspect('equal', adjustable='box')  
        ax.grid(True, linestyle="--", alpha=0.2)
        
    plt.tight_layout()  
    plt.show()  



# ---------- MENÚ DE USUARIO ----------

lista_de_piezas = []  

def mostrar_piezas_agregadas():
    if not lista_de_piezas:
        print("\n(No hay piezas en la lista)\n")
        return
        
    print("\n--- Piezas a Cortar ---")
    print(f"{'#':<3} | {'Cantidad':<10} | {'Ancho':<10} | {'Alto (Largo)':<10}")
    print("-" * 47)
    for i, pieza in enumerate(lista_de_piezas, 1):
        print(f"{i:<3} | {pieza['cantidad']:<10} | {pieza['ancho']:<10} | {pieza['alto']:<10}")
    print("-" * 47)

def agregar_pieza():
    print("\n--- Agregar Nueva Pieza ---")
    try:
        ancho = float(input("Ancho de la pieza (cm): "))
        alto = float(input("Alto (Largo) de la pieza (cm): "))
        cantidad = int(input("¿Cuántas piezas de este tipo?: "))
        
        if ancho <= 0 or alto <= 0 or cantidad <= 0:
            print("Error: Todos los valores deben ser positivos.")
            time.sleep(1)
            return
            
        lista_de_piezas.append({'ancho': ancho, 'alto': alto, 'cantidad': cantidad})
        print(f"¡Agregado! {cantidad}x pieza(s) de {ancho}x{alto} cm.")
        
    except ValueError:
        print("Error: Entrada inválida. Introduce solo números.")
    
    time.sleep(1.5)

def iniciar_optimizacion():
    if not lista_de_piezas:
        print("Error: No hay piezas en la lista. Agrega al menos una pieza.")
        time.sleep(2)
        return
        
    print("\n--- Iniciar Acomodo ---")
    try:
        ancho_plano = float(input("Ancho del PLANO de material (cm): "))
        alto_plano = float(input("Alto (Largo) del PLANO de material (cm): "))
        
        if ancho_plano <= 0 or alto_plano <= 0:
            print("Error: Las dimensiones del plano deben ser positivas.")
            time.sleep(1)
            return
            
        hoja_plano = HojaMaterial(ancho=ancho_plano, alto=alto_plano)
        
        # --- Parámetros del Algoritmo ---
        TAM_POBLACION = 50       # 50
        PROB_CRUCE = 0.6         # 60% - 70% es mejor de probabilidad de cruce
        PROB_MUTACION = 0.01      # 0.01% de probabilidad de mutación 
        NUM_GENERACIONES = 50   # 50 generaciones
        
        ae = AlgoritmoEvolutivo(TAM_POBLACION, PROB_CRUCE, PROB_MUTACION, NUM_GENERACIONES)
        
        ae.cargar_piezas(hoja_plano, lista_de_piezas)
        
        if not ae.tipos_elementos:
            print("Error: No se cargaron elementos.")
            time.sleep(2)
            return
        mejor_solucion = ae.ejecutar()
        
        if mejor_solucion and mejor_solucion.cortado:
            print("Mostrando el mejor patrón de corte encontrado...")
            dibujar_patron_de_corte(hoja_plano, mejor_solucion.cortado)
        else:
            print("No se encontró una solución válida.")

    except ValueError:
        print("Error: Dimensiones inválidas. Introduce solo números.")
        time.sleep(2)

def main():
    opcion = -1
    while opcion != 0:
        os.system("cls" if os.name == "nt" else "clear")
        print("--- Optimizador de Corte de Piezas (Algoritmo Evolutivo) ---")
        mostrar_piezas_agregadas()
        
        print("\nOpciones:")
        print("1. Agregar Pieza")
        print("2. Iniciar Acomodo (Optimizar)")
        print("3. Limpiar lista de piezas")
        print("0. Salir\n")
        
        try:
            opcion = int(input("Elige una opción: "))
        except ValueError:
            opcion = -1
        
        if opcion == 1:
            agregar_pieza()
        elif opcion == 2:
            iniciar_optimizacion()
        elif opcion == 3:
            lista_de_piezas.clear()  
            print("Lista de piezas limpiada.")
            time.sleep(1)
        elif opcion == 0:
            print("\nSaliendo del programa...")
        else:
            print("Opción no válida.")
            time.sleep(1)
            
    time.sleep(1)

if __name__ == "__main__":
    main()