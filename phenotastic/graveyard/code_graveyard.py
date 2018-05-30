
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:01:39 2018

@author: henrik
"""
import vtk
import handy_functions as hf

def new_smooth(A, iters=5, featureAngle=45, passBand=0.001):
  smoother = vtk.vtkWindowedSincPolyDataFilter()
  smoother.SetInputConnection(A.mesh.GetProducerPort())
  smoother.SetNumberOfIterations(iters)
  smoother.BoundarySmoothingOff()
  smoother.FeatureEdgeSmoothingOff()
  smoother.SetFeatureAngle(featureAngle)
  smoother.SetPassBand(passBand)
  smoother.NonManifoldSmoothingOn()
  smoother.NormalizeCoordinatesOn()
  smoother.Update()

  smoothedMapper = vtk.vtkPolyDataMapper()
  smoothedMapper.SetInputConnection(smoother.GetOutputPort())
  return smoothedMapper.GetInput()

def plot_boundary(A):
  featureEdges = vtk.vtkFeatureEdges()
  featureEdges.SetInputConnection(A.mesh.GetProducerPort())
  featureEdges.BoundaryEdgesOn()
  featureEdges.FeatureEdgesOff()
  featureEdges.ManifoldEdgesOff()
  featureEdges.NonManifoldEdgesOff()
  featureEdges.Update()

  edgeMapper = vtk.vtkPolyDataMapper()
  edgeMapper.SetInputConnection(featureEdges.GetOutputPort())
  edgeActor = vtk.vtkActor()
  edgeActor.SetMapper(edgeMapper)

  diskMapper = vtk.vtkPolyDataMapper()
  newPoly = vtk.vtkPolyData()
  newPoly.ShallowCopy(A.mesh)
  diskMapper.SetInputConnection(newPoly.GetProducerPort())
  diskActor = vtk.vtkActor()
  diskActor.SetMapper(diskMapper)

  hf.render_actors([edgeActor, diskActor])
