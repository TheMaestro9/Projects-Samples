package Jowil;

import com.jfoenix.controls.*;
import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.beans.value.ObservableValue;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.geometry.NodeOrientation;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.util.Callback;



import java.io.IOException;
import java.util.*;

import static Jowil.CSVHandler.NOT_AVAILABLE;
import static Jowil.CSVHandler.updateQuestionsChoices;

public class ManualModeController extends Controller{



    @FXML
    TableView table;

    @FXML
    Label tableTitle;

    @FXML
    VBox tableVBox;

    @FXML
    JFXTextField columnSetTextField;

    @FXML
    JFXComboBox columnSetCombo;

    @FXML
    Button addButton;


    @FXML
    ScrollPane scrollPane;

    @FXML
    HBox columnSetHBox;

    @FXML
    Separator midSeparator;

    @FXML
    VBox rightSideVBox;

    @FXML
    Label rightTitleLabel;

    @FXML
    HBox scrollPaneTitlesHBox;

    @FXML
    VBox columnSetsVBox;

    @FXML
    VBox scrollPaneContentVBox;


    @FXML
    Label CSNameLabel;

    @FXML
    Label CSColorLabel;

    @FXML
    Label CSRangeLabel;

    @FXML
    Label CSTypeLabel;

    @FXML
    Button resetButton;

    @FXML
    ImageView addButtonGraphic;

    @FXML
    ImageView resetButtonGraphic;

    static final String PLACE_HOLDER_TEXT="No Column Sets Added";
    Label placeHolder=new Label(isTranslationMode && translations.containsKey(PLACE_HOLDER_TEXT)?translations.get(PLACE_HOLDER_TEXT):PLACE_HOLDER_TEXT);


    ObservableList<ObservableList<StringProperty>> tableContent= FXCollections.observableArrayList();
    private static boolean isManualModeUsedBefore;
    ArrayList<ColumnSet> columnSets=new ArrayList<>();
    int colClicksCount=0;
    ArrayList<String> tableHeaders;
    static final int MAX_ROWS_COUNT=11;
    ArrayList<SimpleStringProperty> colColors;
    int prevClickedCol=NOT_AVAILABLE;
    int firstColIndex=NOT_AVAILABLE;
    int secondColIndex=NOT_AVAILABLE;
    static final String CELL_DEFAULT_COLOR="transparent";
    ColorGenerator colorGen=new ColorGenerator();
    final static String OBJECTIVE_TYPE="Objective Questions Group", SUBJECTIVE_TYPE="Subjective Questions Group",
    ID_TYPE="Student ID",FORM_TYPE="Form Number";
    final static String[] comboOptions={OBJECTIVE_TYPE,SUBJECTIVE_TYPE,ID_TYPE ,FORM_TYPE};
//    int IDStartIndex,IDEndIndex,formIndex;
    int columnSetComboSelectedIndex;
    Controller caller;
    static boolean isIgnoreSavedObjectiveWeights;
    boolean isObjColSetsChanged=false;



    ManualModeController(Controller caller){
        super("ManualMode.fxml","Manual Configuration",1.25,1.25,true,null,false,false,resX*944/1280,0);
        this.caller=caller;

    }

    public static boolean isIsManualModeUsedBefore() {
        return isManualModeUsedBefore;
    }

    public static void setIsManualModeUsedBefore(boolean isManualModeUsedBefore) {
        ManualModeController.isManualModeUsedBefore = isManualModeUsedBefore;
    }

    @Override
    protected void updateSizes() {
        super.updateSizes();


        //left
        tableVBox.setLayoutX(rootWidthToPixels(0.05));
        tableVBox.setLayoutY(rootHeightToPixels(0.04));
        tableVBox.setSpacing(rootHeightToPixels(0.05));
        tableVBox.setPadding(new Insets(0, 0, 0, 0));
        tableVBox.setPrefWidth(rootWidthToPixels(0.55));
        table.setPrefHeight(rootHeightToPixels(0.55));
        columnSetHBox.setSpacing(rootWidthToPixels(0.01));
        columnSetTextField.setPrefWidth(tableVBox.getPrefWidth()*0.25);
        columnSetCombo.setPrefWidth(tableVBox.getPrefWidth()*0.25);
        //columnSetHBox.setPadding(new Insets(rootHeightToPixels(0.05),0,0,0));

        if(isNormalScalingMode) {
            if (rootWidth < minWidth) {
                columnSetTextField.setStyle("-fx-font-size:" + rootWidth * 1.25 * 14 / 1280);
                columnSetCombo.setStyle("-fx-font-size:" + rootWidth * 1.25 * 14 / 1280);
                double scalingFactor = isTranslationMode ? 1.2 : 1.25;
                addButton.setStyle("-fx-font-size:" + rootWidth * scalingFactor * 12 / 1280);
                resetButton.setStyle("-fx-font-size:" + rootWidth * scalingFactor * 12 / 1280);
            } else {
                columnSetTextField.setStyle("-fx-font-size:" + resX * 14 / 1280);
                columnSetCombo.setStyle("-fx-font-size:" + resX * 14 / 1280);
                addButton.setStyle("");
                resetButton.setStyle("");
            }
        }

        double buttImageSize=resX*13/1280;
        addButtonGraphic.setFitWidth(buttImageSize);
        addButtonGraphic.setFitHeight(buttImageSize);
        resetButtonGraphic.setFitWidth(buttImageSize);
        resetButtonGraphic.setFitHeight(buttImageSize);

        midSeparator.setLayoutX(tableVBox.getLayoutY()+tableVBox.getPrefWidth()+rootWidthToPixels(0.05));
        midSeparator.setLayoutY(rootHeightToPixels(0.03));
        midSeparator.setPrefHeight(rootHeightToPixels(0.8));



        //right
        rightSideVBox.setLayoutX((int)(tableVBox.getLayoutX()+tableVBox.getPrefWidth()+rootWidthToPixels(0.05)));
        rightSideVBox.setLayoutY((int)(tableVBox.getLayoutY()));
        rightSideVBox.setSpacing((int)(tableVBox.getSpacing()));
        rightSideVBox.setPadding(tableVBox.getPadding());
        double scrollPaneWidth=(int)(buttonsHbox.getPrefWidth()+buttonsHbox.getLayoutX()-rightSideVBox.getLayoutX());
        double scrollPaneHeight=(int)(table.getPrefHeight());
        scrollPane.setPrefWidth(scrollPaneWidth);
        scrollPane.setPrefHeight(scrollPaneHeight);

        scrollPaneContentVBox.setPadding(new Insets(0, 0, 0, (int)(scrollPaneWidth * 0.02)));
        scrollPaneContentVBox.setSpacing((int)(resYToPixels(0.03)));
        columnSetsVBox.setSpacing((int)(resYToPixels(0.025)));

        placeHolder.setPadding(new Insets(scrollPaneHeight*0.3,scrollPaneWidth*0.4,scrollPaneHeight*0.3,scrollPaneWidth*0.275));


        scrollPaneTitlesHBox.setPadding(new Insets((int)(scrollPaneHeight * 0.05), 0, 0, 0));
        scrollPaneTitlesHBox.setSpacing((int)(scrollPaneWidth * 0.03));


        CSNameLabel.setPrefWidth((int)(scrollPaneWidth*0.21));
        CSTypeLabel.setPrefWidth((int)(scrollPaneWidth*0.21));
        CSColorLabel.setPrefWidth((int)(scrollPaneWidth*0.15));
        CSRangeLabel.setPrefWidth((int)(scrollPaneWidth*0.2));




        for(ColumnSet col:columnSets)
            col.updateSizes(scrollPaneWidth,scrollPaneHeight);

    }

    @Override
    protected void initComponents() {
        tableTitle.setFont(new Font("Arial",headersFontSize));
        rightTitleLabel.setFont(tableTitle.getFont());
        

        scrollPaneTitlesHBox.setStyle("-fx-font-size:"+resX/100+";-fx-font-weight: bold;");
        scrollPane.focusedProperty().addListener((observable, oldValue, newValue) -> {
            if (newValue)
                rootPane.requestFocus();
        });

        scrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        resetButton.setOnMouseClicked(event -> {
            removeAllColSets();
        });

        initTable();
        initColumnSetsVBox();
        initAddButton();
        initColumnSetCombo();
        initNextButton();
        loadExistingColSets();



    }


    @Override
    protected void saveChanges() {

    }

    @Override
    public void startWindow(){
        super.startWindow();
        stage.setOnCloseRequest(event->{
        });
        stage.setY(caller.stage.getY()+15);
    }

    @Override
    protected Controller getNextController() {
        return null;
    }


    @Override
    protected void stabalizeTables(){
        disableTableDrag(table);
    }






    //init functions
    private void initTable(){

        loadTableContents();
        table.setItems(tableContent);

        if(isTranslationMode) //override default behaviour
            table.setNodeOrientation(NodeOrientation.LEFT_TO_RIGHT);

        int colsCount=tableContent.get(0).size();

        colColors=new ArrayList<>();
        for(int i=0;i<colsCount;i++){
            table.getColumns().add(createColumn(i));
            colColors.add(new SimpleStringProperty(CELL_DEFAULT_COLOR));
        }

        TableColumn lastDummy=new TableColumn("   ");
        table.getColumns().add(lastDummy);


        table.getSelectionModel().setCellSelectionEnabled(true);


        Platform.runLater(() -> table.refresh());

        table.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);



    }

    @Override
    protected void initNextButton(){

        nextButton.setText("Save Changes");
        nextButton.setOnMouseClicked(t->{
            rootPane.requestFocus();

            if(!saveColumnSets())
                return;

            isManualModeUsedBefore=true;
            stage.close();
            if(caller instanceof FileConfigController){
                try {
                    Thread.sleep(400);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                caller.stage.hide();
                new GroupsController(caller).startWindow();
            }
            else {
                caller.stage.close();
                new GroupsController(caller.back).startWindow();
            }

        });


    }

    private void loadExistingColSets() {
        int idStart=CSVHandler.getIdentifierColStartIndex();
        int idEnd=CSVHandler.getIdentifierColEndIndex();

        if(idStart!=NOT_AVAILABLE && idEnd!=NOT_AVAILABLE && selectedIdentifierName!=null) {
            String nextColor=colorGen.getNextColor();
            String idName=selectedIdentifierName.replace(MANUAL_MODE_INDICATOR,"");
            columnSets.add(new ColumnSet(idName, nextColor, idStart, idEnd - idStart, this, ID_TYPE));
            for(int i=idStart;i<idEnd;i++)
                colColors.get(i).set(nextColor);
        }

        if(CSVHandler.getFormColIndex()!=NOT_AVAILABLE && selectedFormColName!=null) {
            String nextColor=colorGen.getNextColor();
            String formColName=selectedFormColName.replace(MANUAL_MODE_INDICATOR,"");
            columnSets.add(new ColumnSet(formColName, nextColor, CSVHandler.getFormColIndex(), 1, this, FORM_TYPE));
            colColors.get(CSVHandler.getFormColIndex()).set(nextColor);
        }


        int subjStart=CSVHandler.getSubjStartIndex();
        int subjEnd=CSVHandler.getSubjEndIndex();

        if(subjStart!=NOT_AVAILABLE && subjEnd!=NOT_AVAILABLE) {
            String nextColor=colorGen.getNextColor();
            columnSets.add(new ColumnSet("Subjective", nextColor, subjStart, subjEnd - subjStart, this, SUBJECTIVE_TYPE));
            for(int i=subjStart;i<subjEnd;i++)
                colColors.get(i).set(nextColor);
        }

        int currentGroupStart=CSVHandler.getQuestionsColStartIndex();

        if(CSVHandler.getDetectedGroups()!=null) {
            for (Group group : CSVHandler.getDetectedGroups()) {
                String nextColor = colorGen.getNextColor();
                columnSets.add(new ColumnSet(group.getCleanedName(), nextColor, currentGroupStart, group.getRealQCount(), this, OBJECTIVE_TYPE));
                for (int i = currentGroupStart; i < currentGroupStart + group.getRealQCount(); i++)
                    colColors.get(i).set(nextColor);

                currentGroupStart += group.getRealQCount();
            }
        }

        if(columnSets.size()==0)
            columnSetsVBox.getChildren().setAll(placeHolder);
        else {
            columnSets.sort(new ColumnSetSorter());
            columnSetsVBox.getChildren().setAll(columnSets);
        }

        updateSizes();

    }

    private boolean saveColumnSets(){
        int objStartIndex,objEndIndex,subjStartIndex,subjEndIndex,IDStartIndex,IDEndIndex,formIndex,firstObjCS,lastObjCS;
        objStartIndex=objEndIndex=subjStartIndex=subjEndIndex=IDStartIndex=IDEndIndex=formIndex=firstObjCS=lastObjCS=NOT_AVAILABLE;

        ArrayList<ColumnSet> objColSets=new ArrayList<>();

        String identifierName="ID",formColName="None";

        int index=0;
        for(ColumnSet columnSet:columnSets){
            String type=columnSet.getType();

            if(type.equals(OBJECTIVE_TYPE)){
                if(objStartIndex==NOT_AVAILABLE) { //first objective group
                    objStartIndex = columnSet.getStartIndex();
                    firstObjCS=index;
                }
                objEndIndex=columnSet.getEndIndex();
                lastObjCS=index;
                objColSets.add(columnSet);

            }else if(type.equals(SUBJECTIVE_TYPE)){ //only one col set can exist
                subjStartIndex=columnSet.getStartIndex();
                subjEndIndex=columnSet.getEndIndex();
            }else if(type.equals(ID_TYPE)){ //only one id group col set exist
                identifierName=columnSet.getName();
                IDStartIndex=columnSet.getStartIndex();
                IDEndIndex=columnSet.getEndIndex();
            }else{ //form type must be only one column
                formColName=columnSet.getName();
                formIndex=columnSet.getStartIndex();
            }
            index++;
        }

        //make end indices exclusive
        objEndIndex++;
        subjEndIndex=(subjEndIndex==-1)?-1:subjEndIndex+1;
        IDEndIndex=(IDEndIndex==-1)?-1:IDEndIndex+1;



        if(objStartIndex==-1){
            showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Objective Question Groups Error","" +
                    "You must add at least one objective questions group column set.");
            return false;
        }

        int gapStart;
        if((gapStart=getObjGapStart(objStartIndex,objEndIndex))!=-1){
            String gapStartName=tableHeaders==null?"Column "+(gapStart+1):tableHeaders.get(gapStart);
            showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Objective Question Groups Error","" +
                    constructMessage("Objective questions groups must be consecutive. A gap exists at"," "+gapStartName+"."));
            return false;
        }

        ArrayList<String> unexpectedColSetInfo;
        if((unexpectedColSetInfo=getUnExpectedColSet(firstObjCS,lastObjCS))!=null){
            showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Objective Question Groups Error",constructMessage("" +
                    "Objective questions groups must be consecutive. Column set"," \""+unexpectedColSetInfo.get(0)+"\" ",
                    "of type"," \"" +unexpectedColSetInfo.get(1)+"\" ","cannot be placed between objective column sets."));
            return false;
        }


        if(CSVHandler.getFormsCount()!=1 && formIndex==-1){
            showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Form Number Column Set Error","" +
                    constructMessage(CSVHandler.getFormsCount()+""," forms were detected. A form number column must be added."));
            return false;
        }
        System.out.println("number of forms:"+CSVHandler.getFormsCount());



        if(!saveToCSVHandler(objStartIndex,objEndIndex,subjStartIndex,subjEndIndex,IDStartIndex,IDEndIndex,formIndex,objColSets)){
            isIgnoreSavedObjectiveWeights=false;
            return false;
        }


        if(isObjColSetsChanged){

//            if(isOpenMode){
//                if(!showConfirmationDialog("Reset Questions Choices","Changes was made to objective questions groups. All groups choices will be reset. Are you sure you want to continue?"
//                        ,stage.getOwner()))
//                    return false;
//            }
            CSVHandler.initQuestionsChoices(true);
        }



        Statistics.setIdentifierName(identifierName);
        Controller.selectedIdentifierName=IDStartIndex==NOT_AVAILABLE?"None":identifierName+MANUAL_MODE_INDICATOR;
        Controller.selectedFormColName=formColName.equals("None")?"None":formColName+MANUAL_MODE_INDICATOR;
        return true;
    }


    private void initColumnSetCombo(){
        ObservableList<String> items=FXCollections.observableArrayList();

        for(String s:comboOptions)
            items.add(isTranslationMode&& translations.containsKey(s)?translations.get(s):s);

        columnSetCombo.setItems(items);


        columnSetCombo.setVisibleRowCount(3);
        columnSetCombo.setOnShown(t -> columnSetCombo.getSelectionModel().clearSelection());
        columnSetCombo.setOnHidden(t -> columnSetCombo.getSelectionModel().select(columnSetComboSelectedIndex));

        columnSetCombo.getSelectionModel().selectedIndexProperty().addListener((observable,oldValue,newValue)-> {

            if((Integer)newValue!=-1)
                columnSetComboSelectedIndex=(Integer)newValue;

        });

        columnSetCombo.getSelectionModel().select(0);

        columnSetCombo.setCellFactory(param ->  {
            final ListCell<String> cell = new ListCell<String>() {
                {

                    this.setPrefHeight(rootHeight*0.05);
                }
                @Override public void updateItem(String item,
                                                 boolean empty) {
                    super.updateItem(item, empty);
                    setText(item);
                }
                };
            return cell;
        });

    }

    private void initColumnSetsVBox() {
        placeHolder.setStyle("-fx-font-weight:bold;-fx-text-fill:#727171");
        placeHolder.setFont(new Font(resX*12/1280));
        columnSetsVBox.getChildren().setAll(placeHolder);
    }

    private void initAddButton(){
       // addButton.getStyleClass().add("BlueJFXButton");
        addButton.setOnMouseClicked(t->{
            if(columnSetTextField.getText().isEmpty()) {
                showAlertAndWait(Alert.AlertType.ERROR, stage.getOwner(), "Column Set Error",
                        "Column Set Name cannot be empty.");
                return;
            }

            if(table.getSelectionModel().getSelectedCells().isEmpty()){

                showAlertAndWait(Alert.AlertType.ERROR, stage.getOwner(), "Column Set Error",
                        "No columns selected.");
                return;

            }

            if(isAllowColumnSetAddition())
                addColumnSet();




        });
    }




    //utility functions
    private void loadTableContents() {

        ArrayList<ArrayList<String>> content=new ArrayList<>();

        if(isOpenMode) {
            for (int i = 0; i < Math.min(MAX_ROWS_COUNT + 1, CSVHandler.getSavedResponsesCSV().size()); i++)
                content.add(new ArrayList<>(Arrays.asList(CSVHandler.getSavedResponsesCSV().get(i))));
        }
        else{
            try {
                content=CSVHandler.readResponsesFile(MAX_ROWS_COUNT+1); //+1 to include headers
            } catch (IOException e) {
                e.printStackTrace();
                showAlertAndWait(Alert.AlertType.ERROR, stage.getOwner(), "CSV Reloading Error", "Error in " +
                        "reloading student responses file.");
                return;
            }
        }




        int start=0,end;

        if(CSVHandler.isIsResponsesContainsHeaders()) { //headers exist
            tableHeaders=content.get(0);
            start=1;
        }
        end=Math.min(content.size(),start+MAX_ROWS_COUNT);

        for(int i=start;i<end;i++){
            ObservableList<StringProperty> row= FXCollections.observableArrayList();
            for(int j=0;j<content.get(0).size();j++)
                row.add(new SimpleStringProperty(content.get(i).get(j)));

            tableContent.add(row);
        }
    }


    private TableColumn<ObservableList<StringProperty>, String> createColumn(final int columnIndex) {
        TableColumn<ObservableList<StringProperty>, String> column = new TableColumn<>();
        String title;

        title = tableHeaders==null?"Column "+Integer.toString(columnIndex+1):tableHeaders.get(columnIndex);





        column.setCellFactory((t) -> ManualModeCell.createManualModeCell(columnIndex,table,this));
        column.setSortable(false);
        column.setEditable(false);
        column.setText(title);
        column
                .setCellValueFactory(new Callback<TableColumn.CellDataFeatures<ObservableList<StringProperty>, String>, ObservableValue<String>>() {
                    @Override
                    public ObservableValue<String> call(
                            TableColumn.CellDataFeatures<ObservableList<StringProperty>, String> cellDataFeatures) {
                        ObservableList<StringProperty> values = cellDataFeatures.getValue();
                        if (columnIndex >= values.size()) {
                            return new SimpleStringProperty("");
                        } else {
                            return cellDataFeatures.getValue().get(columnIndex);
                        }
                    }
                });


        return column;
    }

    public boolean selectRequiredRange(int clickedCol){


        if(prevClickedCol!=NOT_AVAILABLE){ //a column was selected before

            int minIndex=Math.min(clickedCol,prevClickedCol);
            int maxIndex=Math.max(clickedCol,prevClickedCol);


            TableColumn minCol=(TableColumn)table.getColumns().get(minIndex);
            TableColumn maxCol=(TableColumn)table.getColumns().get(maxIndex);


            if(isRangeInValid(minIndex,maxIndex)){
                resetTable(true);
                showInvalidSelectionMessage();
                return false;
            }
            else {
                table.getSelectionModel().selectRange(0, minCol, table.getItems().size(), maxCol);
                firstColIndex=prevClickedCol;
                secondColIndex=clickedCol;
            }
        }
        else {

            if(isColumnInvalid(clickedCol)){
                resetTable(true);
                return false;
            }
            else {
                TableColumn col =(TableColumn)table.getColumns().get(clickedCol);
                table.getSelectionModel().selectRange(0, col, table.getItems().size(), col);
                firstColIndex = secondColIndex =clickedCol;
            }
        }

        prevClickedCol=clickedCol;

        colClicksCount++;


        return true;


    }


    private boolean isAllowColumnSetAddition() {

        int type=columnSetComboSelectedIndex;
        String newGroupName=columnSetTextField.getText();


        if(comboOptions[type].equals(FORM_TYPE) && firstColIndex!=secondColIndex){
            showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Column Set Addition Error",
                    constructMessage("A column set of type"," \"",FORM_TYPE,"\" ","cannot have more than one column."));
            return false;
        }

        for (ColumnSet columnSet : columnSets) {

            if(columnSet.getName().equals(newGroupName)){ //check for repeated name
                showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Column Set Addition Error",
                        constructMessage("A column set with the name", " \""+newGroupName+"\" ","already exists. ")); //space b3d already exists 3shan yb2a mo2anas fl targama (see translations)
                return false;
            }

            if (!comboOptions[type].equals(OBJECTIVE_TYPE) && columnSet.getType().equals(comboOptions[type])) { //check for repeated type except if objective group
                boolean isReplaceExisting = showConfirmationDialog("Confirm Column Set Addition",
                        constructMessage((isTranslationMode?"":"A ") + comboOptions[type], " column set already exists. Do you want to replace the existing column set?")
                        , stage.getOwner());

                if (isReplaceExisting) {
                    deleteColumnSet(columnSet);
                    return true;
                } else
                    return false;
            }


        }

        return true;
    }


    private void addColumnSet(){

        int start=Math.min(firstColIndex,secondColIndex);
        int end=Math.max(firstColIndex,secondColIndex);

        if(start==-1)
            return;

        String nextColor=colorGen.getNextColor();

        for(int i=start;i<=end;i++)
            colColors.get(i).set(nextColor);

        resetTable(false);


        String colSetType=comboOptions[columnSetCombo.getSelectionModel().getSelectedIndex()];
        columnSets.add(new ColumnSet(columnSetTextField.getText(),nextColor,start,end-start+1,this,colSetType));
        columnSets.sort(new ColumnSetSorter());

        columnSetsVBox.getChildren().setAll(columnSets);
        updateSizes();

        if(colSetType.equals(OBJECTIVE_TYPE))
            isObjColSetsChanged=true;

        columnSetTextField.clear();

    }

    private boolean isRangeInValid(int minIndex, int maxIndex) {

        for(int i=minIndex;i<=maxIndex;i++){
            if(!colColors.get(i).get().equals(CELL_DEFAULT_COLOR))
                return true;
        }

        return false;

    }

    private boolean isColumnInvalid(int index){
        return !colColors.get(index).get().equals(CELL_DEFAULT_COLOR);
    }


    public void resetTable(boolean isIgnoreClear) {
        this.colClicksCount=0;
        prevClickedCol=NOT_AVAILABLE;

        if(!isIgnoreClear) //avoid clear selection when nothing selected, otherwise ->index out of bounds exception
            table.getSelectionModel().clearSelection();
        this.firstColIndex=NOT_AVAILABLE;
        this.secondColIndex=NOT_AVAILABLE;
    }

    private void showInvalidSelectionMessage(){
        showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Invalid Selection","Selected range "
                +"must not overlap with an existing column set.");
    }

    public void deleteColumnSet(ColumnSet deleted) {
        columnSets.remove(deleted);

        if(columnSets.size()==0)
            columnSetsVBox.getChildren().setAll(placeHolder);
        else
            columnSetsVBox.getChildren().setAll(columnSets);

        String deletedColor=deleted.getColor();
        colorGen.addToAvailable(deletedColor);

        for(SimpleStringProperty colColor:colColors){
            if(colColor.get().equals(deletedColor))
                colColor.set("transparent");
        }

        if(deleted.getType().equals(OBJECTIVE_TYPE))
            isObjColSetsChanged=true;
    }


    private int getObjGapStart(int start, int end) {

        for(int i=start;i<end;i++){
            if(colColors.get(i).get().equals(CELL_DEFAULT_COLOR))
                return i;
        }

        return -1; //gap not found

    }

    private boolean saveToCSVHandler(int objStartIndex, int objEndIndex, int subjStartIndex, int subjEndIndex, int idStartIndex, int idEndIndex, int formIndex, ArrayList<ColumnSet> objColSets) {

        CSVHandler.setQuestionsColStartIndex(objStartIndex);
        CSVHandler.setQuestionsColEndIndex(objEndIndex);

        CSVHandler.setSubjStartIndex(subjStartIndex);
        CSVHandler.setSubjEndIndex(subjEndIndex);
        CSVHandler.setSubjQuestionsCount(subjStartIndex==NOT_AVAILABLE?0:subjEndIndex-subjStartIndex);

        CSVHandler.setFormColIndex(formIndex);

        CSVHandler.setIdentifierColStartIndex(idStartIndex);
        CSVHandler.setIdentifierColEndIndex(idEndIndex);


        try {
            CSVHandler.generateObjectiveGroupsFromColSets(objColSets);
        } catch (CSVHandler.InConsistentAnswerKeyException e) {
            e.printStackTrace(); //maynf3sh yhsl
        } catch (IOException e) {
            showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Answer Key Error","Error in " +
                    "reloading answer key file.");
            return false;
        } catch (CSVHandler.IllFormedCSVException e) {
            e.printStackTrace(); //mynf3sh yhsl
        }


        if(isOpenMode){

            if(!isQuestMode && isObjColSetsChanged) {
                if (CSVHandler.getDetectedQHeaders().size() == Statistics.getQuestionWeights().get(0).size()) {

                    if (showWeightsResetConfirmationDialog())
                        isIgnoreSavedObjectiveWeights = true;
                    else
                        isIgnoreSavedObjectiveWeights = false;

                } else {
                    if (showWeightsWarningDialog())
                        isIgnoreSavedObjectiveWeights = true;
                    else
                        return false;
                }
            }


            try {
                CSVHandler.loadSavedCSV();
                System.out.println("3malt el maslaha");
            } catch (CSVHandler.InvalidFormNumberException e) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                        constructMessage("Error in students responses file: " , e.getMessage(),". Make sure that you have selected a valid form column."));
                return false;
            } catch (CSVHandler.InvalidSubjColumnException e) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                        constructMessage("Error in students responses file: " , e.getMessage()));
                return false;
            }
        }
        else {

            try {
                CSVHandler.loadCsv(CSVHandler.isIsResponsesContainsHeaders());
            } catch (CSVHandler.IllFormedCSVException e) {
                String message=constructMessage("Error in students responses file at row " , e.getRowNumber()+"" ,
                        ". File must contain the same number of columns in all rows.",(e.getRowNumber()==2?" Make sure that the CSV headers have no commas.":""));
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",message);
                return false;
            } catch (IOException e) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                        "Error in reading students responses file.");
                return false;
            } catch (CSVHandler.InvalidFormNumberException | CSVHandler.InvalidSubjColumnException e) {
                showAlert(Alert.AlertType.ERROR, stage.getOwner(), "Students Responses File Error",
                        constructMessage("Error in students responses file: " , e.getMessage()));
                return false;
            }
        }

        for(ColumnSet columnSet:objColSets){  //validate all objective column sets
            String errorMessage="";
            if((errorMessage=CSVHandler.getObjColumnSetErrorMessage(columnSet))!=null){
                showAlertAndWait(Alert.AlertType.ERROR,stage.getOwner(),"Objective Questions Error",errorMessage);
                deleteColumnSet(columnSet);
                return false;
            }
        }

        return true;

    }



    private ArrayList<String> getUnExpectedColSet(int firstObjCS, int lastObjCS) {

        for(int i=firstObjCS;i<=lastObjCS;i++){
            ColumnSet columnSet=columnSets.get(i);
            if(!columnSet.getType().equals(OBJECTIVE_TYPE)){
                ArrayList<String> unExpectedInfo=new ArrayList<>();
                unExpectedInfo.add(columnSet.getName());
                unExpectedInfo.add(columnSet.getType());
                return unExpectedInfo;
            }
        }

        return null;
    }


    private boolean showWeightsWarningDialog() {
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle("Number of Objective Questions Changed");
        alert.setHeaderText(null);
        alert.setContentText("The number of objective questions has changed. All the saved objective weights for this project will be lost.");
        //alert.setOnCloseRequest(t->alert.hide());
        alert.getButtonTypes().setAll(ButtonType.OK,ButtonType.CLOSE);
        Button closeButt=((Button)alert.getDialogPane().lookupButton(ButtonType.CLOSE));
        closeButt.setText("Cancel");

        alert.setGraphic(new ImageView("Images/Error_48px.png"));

        processDialog(alert);
        Optional<ButtonType> result = alert.showAndWait();

        return result.get()==ButtonType.OK;

    }

    private boolean showWeightsResetConfirmationDialog() {

        Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
        alert.setTitle( "Reset Question Weights");
        alert.setHeaderText(null);
        alert.setContentText("Objective questions was just edited. Would you like to reset the existing objective weights?");
        Button okButt=(Button)alert.getDialogPane().lookupButton(ButtonType.OK);
        okButt.setText("Yes, Reset Weights");

        Button cancelButt=(Button)alert.getDialogPane().lookupButton(ButtonType.CANCEL);
        cancelButt.setText("No, Load Saved Weights");


        processDialog(alert);
        alert.initOwner(stage.getOwner());
        Optional<ButtonType> option = alert.showAndWait();

        ;

        return option.get() == ButtonType.OK;


    }

    private void removeAllColSets() {

        boolean isAccept=showConfirmationDialog("Reset Column Sets","Resetting will delete all existing column sets." +
                " Are you sure you want to reset column sets?",stage.getOwner());

        if(isAccept) {
            while (!columnSets.isEmpty())
                deleteColumnSet(columnSets.get(0));
            isObjColSetsChanged=true;
        }


    }


    class ColumnSetSorter implements Comparator<ColumnSet> {

        @Override
        public int compare(ColumnSet o1, ColumnSet o2) {
            return o1.getStartIndex()-o2.getStartIndex();
        }
    }

}
