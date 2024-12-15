import pandas as pd

def process_data(filepath):
    #Lectura del dataset
    data = pd.read_csv(filepath)

    data.rename(columns = {
        'Marital status': 'Estado civil',
        'Application mode': 'Modo de aplicación',
        'Application order': 'Órden de solicitud',
        'Course': 'Carrera',
        'Daytime/evening attendance': 'Turno de asistencia',
        'Previous qualification': 'Titulación previa',
        'Mother\'s occupation': 'Ocupación de la madre',
        'Father\'s occupation': 'Ocupación del padre',
        'Displaced': 'Desplazado',
        'Debtor': 'Deudor',
        'Tuition fees up to date': 'Matrículas al día',
        'Gender': 'Género',
        'Scholarship holder': 'Becario',
        'Age': 'Edad de ingreso',
        'Curricular units 1st sem (credited)': 'Unidades curriculares 1er semestre - acreditadas',
        'Curricular units 2nd sem (credited)': 'Unidades curriculares 2do semestre - acreditadas',
        'Curricular units 1st sem (evaluations)': 'Unidades curriculares 1er semestre - matriculados',
        'Curricular units 2nd sem (evaluations)': 'Unidades curriculares 2do semestre - matriculados',
        'Curricular units 1st sem (approved)': 'Unidades curriculares 1er semestre - aprobadas',
        'Curricular units 1st sem (grade)': 'Unidades curriculares 1er semestre - grado',
        'Curricular units 2nd sem (credited)': 'Unidades curriculares 2do semestre - acreditadas',
        'Curricular units 2nd sem (enrolled)': 'Unidades curriculares 2do semestre - matriculados',
        'Curricular units 2nd sem (evaluations)': 'Unidades curriculares 2do semestre - matriculados',
        'Curricular units 2nd sem (approved)': 'Unidades curriculares 2do semestre - aprobadas',
        'Curricular units 2nd sem (grade)': 'Unidades curriculares 2do semestre - grado',
        'Curricular units 2nd sem (without evaluations)': 'Unidades curriculares 2do semestre - sin evaluaciones',
        'GDP': 'PIB per cápita (USD)'
    }, inplace = True)

    column_mappings = {
        'Estado civil': {
            1: "Soltero",
            2: "Casado",
            3: "Viudo",
            4: "Divorciado",
            5: "Unión de hecho",
            6: "Legalmente separados"
        },
        'Modo de aplicación': {
            1: "1.ª fase: Contingente general",
            2: "Ordenanza N.º 612/93",
            3: "1.ª fase: Cntingente especial - Isla Azores",
            4: "Titulares de otros cursos superiores",
            5: "Ordenanza N.º 854-B/99",
            6: "Estudiante internacional - licenciatura",
            7: "1.ª fase: Contingente especial - Isla Madeira",
            8: "2.ª fase: Contingente general",
            9: "3.ª fase: Contingente general",
            10: "Ordenanza N.º 533-A/99, Plan Diferente",
            11: "Ordenanza N.º 533-A/99, Otra Institución",
            12: "Mayor de 23 años",
            13: "Traslado",
            14: "Cambio de carrera",
            15: "Diplomados en especialización tecnológica",
            16: "Cambio de institución/carrera",
            17: "Diplomados de ciclo corto",
            18: "Cambio de institución/carrera - Internacional"
        },
        'Órden de solicitud': {
            0: "Primera",
            1: "Segunda",
            2: "Tercera",
            3: "Cuarta",
            4: "Quinta",
            5: "Sexta",
            6: "Séptima",
            7: "Octava",
            8: "Novena",
            9: "Última"
        },
        'Carrera': {
            1: "Tecnologías de producción de biocombustibles",
            2: "Animación y Diseño multimedia",
            3: "Servicio social - Turno noche",
            4: "Agronomía",
            5: "Diseño de comunicación",
            6: "Enfermería veterinaria",
            7: "Ingeniería informática",
            8: "Equinicultura",
            9: "Gestión",
            10: "Servicio social",
            11: "Turismo",
            12: "Enfermería",
            13: "Higiene bucal",
            14: "Gerencia de Publicidad y Marketing",
            15: "Periodismo y Comunicación",
            16: "Educación Básica",
            17: "Gestión - Turno noche"
        },
        'Turno de asistencia': {
            1: "Diurno",
            2: "Nocturno"
        },
        'Titulación previa': {
            1: "Educación secundaria",
            2: "Educación superior - Grado de bachiller",
            3: "Educación superior - Graduado",
            4: "Educación superior - Maestría",
            5: "Educación superior - Doctorado",
            6: "Frecuencia de educación superior",
            7: "12° año de escolaridad - no concluido",
            8: "11° año de escolaridad - no concluido",
            9: "Otro - 11° de escolaridad",
            10: "10° de escolaridad",
            11: "10° de escolaridad - no concluido",
            12: "Educación básica 3er ciclo (9º/10º/11º año) o equivalente",
            13: "Educación básica 2º ciclo (6º/7º/8º año) o equivalente",
            14: "Curso de especialización tecnológica",
            15: "Educación superior—licenciatura - 1er ciclo",
            16: "Curso técnico superior profesional",
            17: "Educación superior: maestría - segundo ciclo"
        },
        'Ocupación de la madre': {
            1: "Estudiante",
            2: "Representantes del Poder Legislativo y Órganos Ejecutivos, Directores y Gerentes Ejecutivos",
            3: "Especialistas en Actividades Intelectuales y Científicas",
            4: "Técnicos y Profesiones de Nivel Intermedio",
            5: "Personal administrativo",
            6: "Trabajadores de Servicios Personales, Seguridad y Ventas",
            7: "Agricultores y Trabajadores Calificados en Agricultura, Pesca y Silvicultura",
            8: "Trabajadores Calificados en la Industria, Construcción y Artesanos",
            9: "Operadores de Instalaciones y Máquinas y Trabajadores de Montaje",
            10: "Trabajadores no Calificados",
            11: "Profesiones de las Fuerzas Armadas",
            12: "Otra Situación",
            13: "En blanco",
            14: "Oficiales de las Fuerzas Armadas",
            15: "Sargentos de las Fuerzas Armadas",
            16: "Otro Personal de las Fuerzas Armadas",
            17: "Directores de Servicios Administrativos y Comerciales",
            18: "Directores de Hoteles, Catering, Comercio y otros Servicios",
            19: "Especialistas en Ciencias Físicas, Matemáticas, Ingeniería y Técnicas Afines",
            20: "Profesionales de la Salud",
            21: "Profesores",
            22: "Especialistas en Finanzas, Contabilidad, Organización Administrativa y Relaciones Públicas y Comerciales",
            23: "Técnicos y Profesiones de Nivel Intermedio en Ciencias e Ingeniería",
            24: "Técnicos y Profesionales de Nivel Intermedio de Salud",
            25: "Técnicos de Nivel Intermedio en Servicios Jurídicos, Sociales, Deportivos, Culturales y Similares",
            26: "Técnicos en Tecnologías de la Información y Comunicación",
            27: "Empleados de Oficina, Secretarios en General y Operadores de Procesamiento de Datos",
            28: "Operadores de Datos, Contabilidad, Estadísticas, Servicios Financieros y de Registros",
            29: "Otro Personal de Apoyo Administrativo",
            30: "Trabajadores de Servicios Personales",
            31: "Vendedores",
            32: "Trabajadores de Cuidado Personal y Similares",
            33: "Personal de Servicios de Protección y Seguridad",
            34: "Agricultores Orientados al Mercado y Trabajadores Calificados en Producción Agrícola y Ganadera",
            35: "Agricultores, Criadores de Ganado, Pescadores, Cazadores y Recolectores de Subsistencia",
            36: "Trabajadores Calificados en la Construcción y Similares, Excepto Electricistas",
            37: "Trabajadores Calificados en Metalurgia, Metalistería y Similares",
            38: "Trabajadores Calificados en Electricidad y Electrónica",
            39: "Trabajadores en Procesamiento de Alimentos, Carpintería, Industria Textil y Otras Industrias y Artesanías",
            40: "Operadores de Plantas y Máquinas Fijas",
            41: "Trabajadores de Ensamblaje",
            42: "Conductores de Vehículos y Operadores de Equipos Móviles",
            43: "Trabajadores No Calificados en Agricultura, Producción Animal, Pesca y Silvicultura",
            44: "Trabajadores No Calificados en la Industria Extractiva, Construcción, Manufactura y Transporte",
            45: "Asistentes de Preparación de Comidas",
            46: "Vendedores Ambulantes, Excepto Alimentos, y Proveedores de Servicios en la Calle"
        },
        'Ocupación del padre': {
            1: "Estudiante",
            2: "Representantes del Poder Legislativo y Órganos Ejecutivos, Directores y Gerentes Ejecutivos",
            3: "Especialistas en Actividades Intelectuales y Científicas",
            4: "Técnicos y Profesiones de Nivel Intermedio",
            5: "Personal administrativo",
            6: "Trabajadores de Servicios Personales, Seguridad y Ventas",
            7: "Agricultores y Trabajadores Calificados en Agricultura, Pesca y Silvicultura",
            8: "Trabajadores Calificados en la Industria, Construcción y Artesanos",
            9: "Operadores de Instalaciones y Máquinas y Trabajadores de Montaje",
            10: "Trabajadores no Calificados",
            11: "Profesiones de las Fuerzas Armadas",
            12: "Otra Situación",
            13: "En blanco",
            14: "Oficiales de las Fuerzas Armadas",
            15: "Sargentos de las Fuerzas Armadas",
            16: "Otro Personal de las Fuerzas Armadas",
            17: "Directores de Servicios Administrativos y Comerciales",
            18: "Directores de Hoteles, Catering, Comercio y otros Servicios",
            19: "Especialistas en Ciencias Físicas, Matemáticas, Ingeniería y Técnicas Afines",
            20: "Profesionales de la Salud",
            21: "Profesores",
            22: "Especialistas en Finanzas, Contabilidad, Organización Administrativa y Relaciones Públicas y Comerciales",
            23: "Técnicos y Profesiones de Nivel Intermedio en Ciencias e Ingeniería",
            24: "Técnicos y Profesionales de Nivel Intermedio de Salud",
            25: "Técnicos de Nivel Intermedio en Servicios Jurídicos, Sociales, Deportivos, Culturales y Similares",
            26: "Técnicos en Tecnologías de la Información y Comunicación",
            27: "Empleados de Oficina, Secretarios en General y Operadores de Procesamiento de Datos",
            28: "Operadores de Datos, Contabilidad, Estadísticas, Servicios Financieros y de Registros",
            29: "Otro Personal de Apoyo Administrativo",
            30: "Trabajadores de Servicios Personales",
            31: "Vendedores",
            32: "Trabajadores de Cuidado Personal y Similares",
            33: "Personal de Servicios de Protección y Seguridad",
            34: "Agricultores Orientados al Mercado y Trabajadores Calificados en Producción Agrícola y Ganadera",
            35: "Agricultores, Criadores de Ganado, Pescadores, Cazadores y Recolectores de Subsistencia",
            36: "Trabajadores Calificados en la Construcción y Similares, Excepto Electricistas",
            37: "Trabajadores Calificados en Metalurgia, Metalistería y Similares",
            38: "Trabajadores Calificados en Electricidad y Electrónica",
            39: "Trabajadores en Procesamiento de Alimentos, Carpintería, Industria Textil y Otras Industrias y Artesanías",
            40: "Operadores de Plantas y Máquinas Fijas",
            41: "Trabajadores de Ensamblaje",
            42: "Conductores de Vehículos y Operadores de Equipos Móviles",
            43: "Trabajadores No Calificados en Agricultura, Producción Animal, Pesca y Silvicultura",
            44: "Trabajadores No Calificados en la Industria Extractiva, Construcción, Manufactura y Transporte",
            45: "Asistentes de Preparación de Comidas",
            46: "Vendedores Ambulantes, Excepto Alimentos, y Proveedores de Servicios en la Calle"
        },
        'Desplazado': {
            1: "Sí",
            0: "No"
        },
        'Deudor': {
            1: "Sí",
            0: "No"
        },
        'Matrículas al día': {
            1: "Sí",
            0: "No"
        },
        'Género': {
            1: "Masculino",
            0: "Femenino"
        },
        'Becario': {
            1: "Sí",
            0: "No"
        }
    }

    data.isnull().sum()/len(data)*100

    #Conversión de variable objetivo
    '''data['Target'] = data['Target'].map({
        'Dropout': 0,
        'Enrolled': 1,
        'Graduate': 2
    })'''

    for column, mapping in column_mappings.items():
        if column in data.columns:
            data[column] = data[column].replace(mapping)

    return data
#--------------------------------------------------------------------------------------------

def copy_data(data):
    new_data = data.copy()
    new_data = new_data.drop(columns=['Nationality', 'Mother\'s qualification',
                                        'Father\'s qualification', 'Educational special needs',
                                        'International', 'Curricular units 1st sem (without evaluations)',
                                        'Unemployment rate', 'Inflation rate'], axis=1)
    return new_data