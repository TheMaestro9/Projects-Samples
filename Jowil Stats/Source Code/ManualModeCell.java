package Jowil;


import javafx.geometry.Pos;
import javafx.scene.control.TableCell;
import javafx.scene.control.TableView;
import javafx.scene.input.MouseEvent;



public class ManualModeCell<S, T> extends TableCell<S, T> {

    int columnIndex;
    TableView myTable;
    ManualModeController controller;
    boolean isMouseClicked=false; //used to prevent false alarm on initial focusing of column 0
    boolean isIgnoreSelection=false;
    boolean isForceFocus=false;

    ManualModeCell(int columnIndex,TableView myTable,ManualModeController controller){


        this.columnIndex=columnIndex;
        this.myTable=myTable;
        this.setPrefHeight(myTable.getPrefHeight()/(ManualModeController.MAX_ROWS_COUNT+2));
        this.setAlignment(Pos.CENTER);
        this.controller=controller;



        initListeners();

        String color=controller.colColors.get(columnIndex).get();
        if(!color.equals("transparent"))  //in order to load existing col sets properly
            this.setStyle("-fx-background-color:"+color);

        controller.colColors.get(columnIndex).addListener((obs,oldValue,newValue)->{
            if(newValue.equals("transparent"))
                this.setStyle("");
            else
                this.setStyle("-fx-background-color:"+newValue);
        });

    }


    private void initListeners() {



        itemProperty().addListener((obx, oldItem, newItem) -> {
            if (newItem == null) {
                setText(null);
            } else {
                setText(newItem.toString());
            }
        });



        this.addEventFilter(MouseEvent.MOUSE_PRESSED, (event) -> {

            if(isForceFocus) {
                this.setFocused(true);
                this.isForceFocus=false;
            }

            if(event.isShortcutDown() || event.isShiftDown() || event.isControlDown()) //ignore if those buttons down
                event.consume();

            if(controller.colClicksCount==2){ //third click in a row
                controller.resetTable(false);
                event.consume();
            }
            isMouseClicked=true;

        });

        this.focusedProperty().addListener((obs, oldVal, newVal) -> {


            if(newVal && isMouseClicked)
                this.isIgnoreSelection = !controller.selectRequiredRange(columnIndex);




        });

        this.selectedProperty().addListener((obs, oldVal, newVal) -> {


            if( newVal && this.isIgnoreSelection) {
                this.updateSelected(false);
                this.isIgnoreSelection = false;
                this.isForceFocus=true; //workaround for probably a bug in javafx
                this.setFocused(false); //to prevent focusing
            }



        });
    }

    public static <S> ManualModeCell<S, String> createManualModeCell(int columnIndex, TableView myTable,ManualModeController controller) {
        return new ManualModeCell<S, String>(columnIndex,myTable,controller);
    }



}
