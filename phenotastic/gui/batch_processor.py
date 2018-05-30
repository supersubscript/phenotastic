############################################################################
#
# File author(s): Max BRAMBACH <max.brambach.0065@student.lu.se>
# Copyright (c) 2017, Max Brambach
# All rights reserved.
# * Redistribution and use in source and binary forms, with or without
# * modification, are not permitted.
#
############################################################################

import sys, os
import numpy as np
from PyQt4 import QtCore, QtGui, uic
import Meristem_Phenotyper_3D as ap
from tissueviewer.tvtiff import tiffread
 
qtCreatorFile = "batch_processor_gui.ui"
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class BatchProcessor(QtGui.QMainWindow, Ui_MainWindow):
    """Main window of Mersitem Phneotyper 3D batch processor.
        
    A graphical user interface for the Meristem Phenotyper 3D package.
    It uses the same parameters as the implementation into Yassin Refahis 
    Tissueviewer. 
    """
    def __init__(self):
        """Builder
        
        Import GUI file and add connections and tooltips."""
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.button_start.clicked.connect(self.process)
        self.pushButton_delete_file.clicked.connect(self.remove_file_from_list)
        self.toolButton_save_location.clicked.connect(self.selectFile)
        self.setAcceptDrops(True)
        self.files = []
        self.actionAbout.triggered.connect(self.open_about_window)
        self.cf_type_input.setStatusTip('Specify the type of contour fit. Threshold is fast, but potentially not robust; Active Contours is robust but slow.')
        self.cf_weight_factor_input.setStatusTip('Specify weight factor. Higher values make the algorithm more sensitive to holes on the surface -> contour rougher.')
        self.cf_smooth_input.setStatusTip('Specify the number of smoothing iterations per contour fit step. Higher values give smoother surface and reduce noise/artifacts in image, but take longer time to compute.')
        self.mesh_smooth_input.setStatusTip('Specify the number of iterations for the mesh smoothing. Higher values give smoother surface.')
        self.relaxation_factor_input.setStatusTip('Specify the relaxation factor for the mesh smoothing. Higher values give smoother surface.')
        self.curvature_type_threshold.setStatusTip('Specify used method to calculate curvature. Mean: robust but not too sensitive for small primordia; Max: not very robust but very sensitive; Gaussian: in between Mean and Max.')
        self.curvature_threshold_input.setStatusTip('Specify threshold for mesh slicing. 0 is very robust. Small deviations from 0 can be used to extract more primordia. Note: Specified value is divided by 100.')
        self.min_feature_size_percent_input.setStatusTip('Specify minimum size (in percent) of object in sliced mesh to be recognised as primordium or meristem.')
        self.cf_type_input.setToolTip('Specify the type of contour fit. \nThreshold is fast, but potentially not robust; \nActive Contours is robust but slow.')
        self.cf_weight_factor_input.setToolTip('Specify weight factor. \nHigher values make the algorithm more sensitive to holes on the surface \n-> contour rougher.')
        self.cf_smooth_input.setToolTip('Specify the number of smoothing iterations per contour fit step. \nHigher values give smoother surface and reduce noise/artifacts in image, but take longer time to compute.')
        self.mesh_smooth_input.setToolTip('Specify the number of iterations for the mesh smoothing. \nHigher values give smoother surface.')
        self.relaxation_factor_input.setToolTip('Specify the relaxation factor for the mesh smoothing. \nHigher values give smoother surface.')
        self.curvature_type_threshold.setToolTip('Specify used method to calculate curvature. \nMean: robust but not too sensitive for small primordia; \nMax: not very robust but very sensitive; \nGaussian: in between Mean and Max.')
        self.curvature_threshold_input.setToolTip('Specify threshold for mesh slicing. \n0 is very robust. \nSmall deviations from 0 can be used to extract more primordia. \nNote: Specified value is divided by 100.')
        self.min_feature_size_percent_input.setToolTip('Specify minimum size (in percent) of object in sliced mesh to be recognised as primordium or meristem.')
                
        
    def dragEnterEvent(self, event):
        """Manage what happens, if a file is dragged onto the GUI.       
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile().toLocal8Bit().data()
            if os.path.isfile(path):
                base = os.path.basename(path)
                if os.path.splitext(base)[1] == '.tif':
                    self.files.append(path)
                    self.listWidget_filenames.clear()
                    self.listWidget_filenames.addItems(self.files)
                    self.pushButton_delete_file.setEnabled(True)
        
    def process(self):
        cont_types = ['Threshold','ACWE']
        curv_types = ['mean','gauss','max']
        save_loc = self.lineEdit_save_location.text()
        contour_type = cont_types[self.cf_type_input.currentIndex()]
        weight_factor = self.cf_weight_factor_input.value()
        smooth_contour = self.cf_smooth_input.value()
        smooth_mesh = self.mesh_smooth_input.value()
        relax_mesh = self.relaxation_factor_input.value()
        curvature_type = curv_types[self.curvature_type_threshold.currentIndex()]
        curvature_threshold = self.curvature_threshold_input.value()/100.
        min_features_size = self.min_feature_size_percent_input.value()
        fit_paraboloid = self.checkBox_paraboloid_fit.isChecked()
        self.proThread = processThread(contour_type,
                                         weight_factor,
                                         smooth_contour,
                                         smooth_mesh,
                                         relax_mesh,
                                         curvature_type,
                                         curvature_threshold,
                                         min_features_size,
                                         fit_paraboloid,
                                         save_loc,
                                         self.files)
        self.connect(self.proThread, self.proThread.status, self.progress)
        self.proThread.start()
        self.proThread.finished.connect(self.processDone)
        
    def processDone(self):
        self.progressBar_total.setValue(100.)
        
    def progress(self, status):
        self.progressBar_total.setValue((status)/float(len(self.files))*100.)
    
    def remove_file_from_list(self):
        for SelectedItem in self.listWidget_filenames.selectedItems():
            del self.files[self.listWidget_filenames.currentRow()]
            self.listWidget_filenames.takeItem(self.listWidget_filenames.row(SelectedItem))
    
    def selectFile(self):
        self.lineEdit_save_location.setText(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
     
    def open_about_window(self):
        QtGui.QMessageBox.information(self, 'About Meristem Phenotyper 3D - Batch Processor',
                                            """File author(s): Max BRAMBACH <max.brambach.0065@student.lu.se>
Copyright (c) 2017, Max Brambach
All rights reserved.
* Redistribution and use in source and binary forms, with or without modification, are not permitted""", 
                                        QtGui.QMessageBox.Ok)
        
class processThread(QtCore.QThread):
    def __init__(self, 
                 contour_type,
                 weight_factor,
                 smooth_contour,
                 smooth_mesh,
                 relax_mesh,
                 curvature_type,
                 curvature_threshold,
                 min_features_size,
                 fit_paraboloid,
                 save_loc,
                 file_loc):
        QtCore.QThread.__init__(self)
        self.contour_type = contour_type
        self.weight_factor = weight_factor
        self.smooth_contour = smooth_contour
        self.smooth_mesh = smooth_mesh
        self.relax_mesh = relax_mesh
        self.curvature_type = curvature_type
        self.curvature_threshold = curvature_threshold
        self.min_features_size = min_features_size
        self.fit_paraboloid = fit_paraboloid
        self.save_loc = save_loc
        self.file_loc = file_loc
        self.status = QtCore.SIGNAL("signal")
    
    def run(self):
        for i in range(len(self.file_loc)):
            A = ap.AutoPhenotype()
            self.emit(self.status, 0.1+i)
            A.data = ap.readTiff(self.file_loc[i])
            if self.contour_type == 'Threshold':
                A.contour_fit_threshold(self.weight_factor, self.smooth_contour)
            elif self.contour_type == 'ACWE':
                A.contour_fit(self.weight_factor,self.smooth_contour)
            self.emit(self.status, 0.2+i)
            A.clear('data')
            A.mesh_conversion()
            self.emit(self.status, 0.3+i)
            A.clean_mesh()
            A.clear('contour')
            self.emit(self.status, 0.4+i)
            A.smooth_mesh(self.smooth_mesh, self.relax_mesh)
            self.emit(self.status, 0.5+i)
            A.curvature_slice(self.curvature_threshold,self.curvature_type)
            self.emit(self.status, 0.6+i)
            A.feature_extraction(self.min_features_size)
            self.emit(self.status, 0.7+i)
            if len(A.features) == 0:
                print 'meristem not found'
            A.clear('mesh')
            A.sphere_fit()
            self.emit(self.status, 0.8+i)
            A.sphere_evaluation()
            if self.fit_paraboloid == True:
                A.paraboloid_fit_mersitem()
            self.emit(self.status, 0.9+i)
            A.clear('features')
            A.save_results(self.save_loc+'/'+os.path.splitext(os.path.basename(self.file_loc[i]))[0])
            self.emit(self.status, i)
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = BatchProcessor()
    window.show()
    sys.exit(app.exec_())