import os
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import copy
import numpy as np 
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



# CUCKOO SEARCH 

class AlgoritmoCuckooSearch:
    
    def __init__(self, n_nidos: int, prob_descubrimiento: float, num_generaciones: int):
            
        self.n_nidos = n_nidos
        self.pa = prob_descubrimiento  
        self.num_generaciones = num_generaciones
        
        self.hoja_base: HojaMaterial = None
        self.tipos_elementos: List[TipoElemento] = []
        self.nidos: List[Individuo] = [] # La población de nidos
        self.mapa_tipos: Dict[int, TipoElemento] = {}
        self.mejor_cuckoo_global: Individuo = None

    def cargar_piezas(self, hoja: HojaMaterial, piezas_a_cargar: List[Dict]):
        self.hoja_base = hoja
        self.tipos_elementos = []
        self.mapa_tipos = {}
        
        id_counter = 0
        for pieza_dict in piezas_a_cargar:
            tipo = TipoElemento(
                id=id_counter,
                ancho=pieza_dict['ancho'],
                alto=pieza_dict['alto'],
                cantidad=pieza_dict['cantidad'],
                nombre=f"Pieza {id_counter+1}"
            )
            self.tipos_elementos.append(tipo)
            self.mapa_tipos[id_counter] = tipo
            id_counter += 1

        print(f"Plano base: {self.hoja_base.ancho} x {self.hoja_base.alto}")
        print(f"Cargados {len(self.tipos_elementos)} tipos.")

    def obtener_paso_levy(self, beta=1.5) -> int:
        
        # Calcula la longitud del salto usando la distribución de Lévy.
        # Retorna un entero que indica cuántos cambios (swaps) hacer.
        
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)
        
        step = u / (abs(v) ** (1 / beta))

        magnitud = int(abs(step) * 2) 
        
        if magnitud < 1: magnitud = 1 # Mínimo 1 cambio
        if magnitud > 10: magnitud = 10 # Límite para no destruir todo
        return magnitud

    def aplicar_vuelo_levy(self, genes_originales: List[int]) -> List[int]:

        nuevos_genes = genes_originales[:]
        n = len(nuevos_genes)
        if n < 2: return nuevos_genes

        magnitud = self.obtener_paso_levy()
        
        if magnitud <= 2:
            for _ in range(magnitud):
                i, j = random.sample(range(n), 2)
                nuevos_genes[i], nuevos_genes[j] = nuevos_genes[j], nuevos_genes[i]
        
        else:
            largo_bloque = min(n, magnitud * 2) 
            inicio = random.randint(0, n - largo_bloque)
            bloque = nuevos_genes[inicio : inicio + largo_bloque]
            random.shuffle(bloque)
            nuevos_genes[inicio : inicio + largo_bloque] = bloque
            
        return nuevos_genes

    def generar_nidos_iniciales(self):
        print("Inicializando nidos de Cuckoos...")
        indices_tipos = [tipo.id for tipo in self.tipos_elementos]
        self.nidos = []
        
        for _ in range(self.n_nidos):
            genes = random.sample(indices_tipos, len(indices_tipos))
            self.nidos.append(Individuo(genes=genes))

    def vaciar_nido(self) -> Individuo:
        indices_tipos = [tipo.id for tipo in self.tipos_elementos]
        genes = random.sample(indices_tipos, len(indices_tipos))
        return Individuo(genes=genes)

    def calcular_aptitud(self, individuo: Individuo):
        # 1. Decodificar genes
        elementos_a_colocar: List[Elemento] = []
        for id_tipo in individuo.genes:
            tipo = self.mapa_tipos[id_tipo]
            for _ in range(tipo.cantidad):
                elementos_a_colocar.append(Elemento(tipo.id, tipo.ancho, tipo.alto, tipo.nombre))

        patron_cortado: TCortado = [THojaCortada()]
        W_HOJA = self.hoja_base.ancho
        L_HOJA = self.hoja_base.alto

        # --- BUCLE PRINCIPAL ---
        for elem in elementos_a_colocar:
            opciones = [] 

            candidatos_rotacion = [(elem.ancho, elem.alto, False), (elem.alto, elem.ancho, True)]
            if elem.ancho == elem.alto: candidatos_rotacion = [candidatos_rotacion[0]]

            for ancho, alto, rotada in candidatos_rotacion:
                if ancho > W_HOJA or alto > L_HOJA: continue 

                for i_h, hoja in enumerate(patron_cortado):
                    for i_t, tira in enumerate(hoja.tiras):
                        
                        for i_p, pila in enumerate(tira.pilas):
                            if pila.ancho == ancho and (pila.alto_usado + alto <= tira.alto):
                                opciones.append((0, i_h, i_t, i_p, rotada))
                        
                        if (tira.ancho_usado + ancho <= W_HOJA):
                            if abs(tira.alto - alto) < 0.01: 
                                opciones.append((10, i_h, i_t, -1, rotada))
                            
                            elif alto < tira.alto:
                                desperdicio = tira.alto - alto
                                score_ajustado = 20 + (desperdicio / L_HOJA)
                                opciones.append((score_ajustado, i_h, i_t, -1, rotada))

                    if hoja.alto_usado + alto <= L_HOJA:
                        score = 30
                        opciones.append((score, i_h, -1, -1, rotada))
            
            if not opciones:
                for _, alto, rotada in candidatos_rotacion:
                     if alto <= L_HOJA: 
                        opciones.append((40, -1, -1, -1, rotada))

            if not opciones: continue 

            mejor = sorted(opciones, key=lambda x: x[0])[0]
            score, i_h, i_t, i_p, es_rotada = mejor
            
            ancho_f = elem.alto if es_rotada else elem.ancho
            alto_f = elem.ancho if es_rotada else elem.alto
            nombre_f = elem.nombre + " (R)" if es_rotada else elem.nombre
            elem_obj = Elemento(elem.id_tipo, ancho_f, alto_f, nombre_f)

            if score == 40: 
                nueva_hoja = THojaCortada()
                nueva_tira = TTira(alto=alto_f, ancho_usado=ancho_f)
                nueva_pila = TPila(ancho=ancho_f, alto_usado=alto_f, elementos=[elem_obj])
                nueva_tira.pilas.append(nueva_pila)
                nueva_hoja.tiras.append(nueva_tira)
                nueva_hoja.alto_usado = alto_f
                patron_cortado.append(nueva_hoja)
            
            elif int(score) == 30: 
                hoja = patron_cortado[i_h]
                nueva_tira = TTira(alto=alto_f, ancho_usado=ancho_f)
                nueva_pila = TPila(ancho=ancho_f, alto_usado=alto_f, elementos=[elem_obj])
                nueva_tira.pilas.append(nueva_pila)
                hoja.tiras.append(nueva_tira)
                hoja.alto_usado += alto_f
            
            elif int(score) == 10 or int(score) == 20:
                tira = patron_cortado[i_h].tiras[i_t]
                nueva_pila = TPila(ancho=ancho_f, alto_usado=alto_f, elementos=[elem_obj])
                tira.pilas.append(nueva_pila)
                tira.ancho_usado += ancho_f
            
            elif score == 0: # 
                pila = patron_cortado[i_h].tiras[i_t].pilas[i_p]
                pila.elementos.append(elem_obj)
                pila.alto_usado += alto_f

        individuo.cortado = patron_cortado
        
        # --- CÁLCULO DE APTITUD ---
        area_total_plano = W_HOJA * L_HOJA
        area_ocupada_total = 0
        numero_hojas = len(patron_cortado)
        for hoja in patron_cortado:
            for tira in hoja.tiras:
                for pila in tira.pilas:
                    for e in pila.elementos:
                        area_ocupada_total += (e.ancho * e.alto)
        
        area_desperdiciada = (numero_hojas * area_total_plano) - area_ocupada_total
        individuo.aptitud = area_desperdiciada
   
   
    # --- CICLO PRINCIPAL CUCKOO SEARCH ---

    def ejecutar(self) -> Individuo:
        if not self.hoja_base or not self.tipos_elementos:
            print("Error: Datos no cargados.")
            return None

        # 1. Inicialización
        self.generar_nidos_iniciales()
        for nido in self.nidos:
            self.calcular_aptitud(nido)
        
        # Ordenar para encontrar el mejor inicial
        self.nidos.sort()
        self.mejor_cuckoo_global = copy.deepcopy(self.nidos[0])
        print(f"Gen 0: Mejor Aptitud = {self.mejor_cuckoo_global.aptitud:.4f}")

        # 2. Ciclo Evolutivo
        for gen in range(1, self.num_generaciones + 1):
            
            # --- FASE 1: Obtener un Cuckoo (Vuelo de Lévy) ---
            # iteramos sobre toda la población para generar nuevos huevos.
            
            for i in range(self.n_nidos):
                idx_nido_host = random.randint(0, self.n_nidos - 1)

                nuevos_genes = self.aplicar_vuelo_levy(self.nidos[i].genes)
                nuevo_cuckoo = Individuo(genes=nuevos_genes)
                self.calcular_aptitud(nuevo_cuckoo)
                
                if nuevo_cuckoo.aptitud < self.nidos[idx_nido_host].aptitud:
                    self.nidos[idx_nido_host] = nuevo_cuckoo 
            
            mejor_gen_actual = min(self.nidos, key=lambda x: x.aptitud)
            if mejor_gen_actual.aptitud < self.mejor_cuckoo_global.aptitud:
                self.mejor_cuckoo_global = copy.deepcopy(mejor_gen_actual)

            # --- FASE 2: Abandono de Nidos ---
            # Ordenamos los nidos de mejor a peor
            self.nidos.sort()
            
            num_a_abandonar = int(self.n_nidos * self.pa)
            
            for i in range(self.n_nidos - num_a_abandonar, self.n_nidos):
                self.nidos[i] = self.vaciar_nido()
                self.calcular_aptitud(self.nidos[i])
            
            self.nidos.sort()
            if self.nidos[0].aptitud < self.mejor_cuckoo_global.aptitud:
                self.mejor_cuckoo_global = copy.deepcopy(self.nidos[0])

            if gen % 10 == 0 or gen == self.num_generaciones:
                print(f"Gen {gen}: Mejor Aptitud = {self.mejor_cuckoo_global.aptitud:.4f}")

        print("\n--- Búsqueda Cuckoo Terminada ---")
        print(f"Mejor solución: {self.mejor_cuckoo_global.aptitud:.4f}")
        return self.mejor_cuckoo_global
    

def dibujar_patron_de_corte(hoja_base, patron):
    num_hojas = len(patron)
    fig, axs = plt.subplots(1, num_hojas, figsize=(7 * num_hojas, 10))
    if num_hojas == 1: axs = [axs]
    for i, hoja in enumerate(patron):
        ax = axs[i]
        ax.set_title(f"Cuckoo Search - Plano {i+1}")
        ax.set_xlim(0, hoja_base.ancho); ax.set_ylim(0, hoja_base.alto)
        ax.add_patch(patches.Rectangle((0,0), hoja_base.ancho, hoja_base.alto, fill=False))
        
        y_c = 0
        for tira in hoja.tiras:
            ax.add_patch(patches.Rectangle((0, y_c), hoja_base.ancho, tira.alto, linestyle='--', edgecolor='blue', fill=False))
            x_c = 0
            for pila in tira.pilas:
                y_e = y_c
                for elem in pila.elementos:
                    color = (random.random(), random.random(), random.random())
                    ax.add_patch(patches.Rectangle((x_c, y_e), elem.ancho, elem.alto, facecolor=color, edgecolor='black', alpha=0.8))
                    if elem.ancho > 5 and elem.alto > 5:
                        ax.text(x_c+elem.ancho/2, y_e+elem.alto/2, f"{elem.nombre}", ha='center', fontsize=7)
                    y_e += elem.alto
                x_c += pila.ancho
            y_c += tira.alto
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# --- MENÚ PRINCIPAL ---
lista_de_piezas = []

def main():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("--- Optimizador CUCKOO SEARCH ---")
        print(f"Piezas agregadas: {len(lista_de_piezas)}")
        print("1. Agregar Pieza")
        print("2. Ejecutar Cuckoo Search")
        print("3. Limpiar")
        print("0. Salir")
        
        op = input("Opción: ")
        
        if op == '1':
            try:
                w = float(input("Ancho de la pieza (cm): "))
                h = float(input("Alto (Largo) de la pieza (cm): "))
                c = int(input("¿Cuántas piezas de este tipo?: "))
                lista_de_piezas.append({'ancho': w, 'alto': h, 'cantidad': c})
            except: pass
        
        elif op == '2':
            if not lista_de_piezas: continue
            try:
                W = float(input("Ancho Plano: "))
                H = float(input("Alto Plano: "))
                
                # PARÁMETROS CUCKOO SEARCH
                N_NIDOS = 25          # Cantidad de cuckoos
                PROB_PA = 0.25        # % de nidos se abandonan
                MAX_GEN = 50          # Generaciones
                
                cs = AlgoritmoCuckooSearch(N_NIDOS, PROB_PA, MAX_GEN)
                cs.cargar_piezas(HojaMaterial(W, H), lista_de_piezas)
                mejor = cs.ejecutar()
                
                if mejor: dibujar_patron_de_corte(HojaMaterial(W, H), mejor.cortado)
                input("Presiona Enter...")
            except ValueError: pass
            
        elif op == '3': lista_de_piezas.clear()
        elif op == '0': break

if __name__ == "__main__":
    main()