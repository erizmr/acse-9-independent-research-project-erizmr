import vtk
from vtk.util.numpy_support import vtk_to_numpy
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os


def read_vtk(file_name, filetype="vtu", *args):
    """Read the data in vtk/vtu/vtp files
    Return a dictionary containing 'mesh' and other physics field data in numpy form
    """
    
    reader = vtk.vtkXMLUnstructuredGridReader()
    if filetype =="vtp":
        reader = vtk.vtkXMLPolyDataReader()
    
    reader.SetFileName(file_name)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()
    
    # Get the field data
    vtk_array_dict = dict()
    numpy_array_dict = dict()
    
    field_names = [item for item in args]

    while isinstance(field_names[0], list):
        field_names = [item for item in field_names[0]]
    
    # Get the mesh data
    nodes_vtk_array= reader.GetOutput().GetPoints().GetData()
    numpy_array_dict['mesh'] = vtk_to_numpy(nodes_vtk_array)
    
    for name in field_names:
        vtk_array_dict[name] = output.GetPointData().GetArray(name)
        numpy_array_dict[name] = vtk_to_numpy(output.GetPointData().GetArray(name))

    return numpy_array_dict