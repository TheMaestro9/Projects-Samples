package Jowil;

import com.jfoenix.controls.*;
import com.jfoenix.controls.cells.editors.TextFieldEditorBuilder;
import com.jfoenix.controls.cells.editors.base.GenericEditableTreeTableCell;
import com.jfoenix.controls.datamodels.treetable.RecursiveTreeObject;
import com.sun.javafx.scene.control.skin.TableHeaderRow;
import com.sun.javafx.scene.control.skin.TableViewSkinBase;
import de.jensd.fx.glyphs.fontawesome.FontAwesomeIconView;
import javafx.application.Application;
import javafx.beans.binding.Bindings;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ListChangeListener;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.geometry.HPos;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.control.cell.CheckBoxTreeCell;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.control.cell.TextFieldTableCell;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.*;
import javafx.scene.text.Font;
import org.apache.commons.math3.stat.inference.TestUtils;


import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.util.*;
import java.util.function.Predicate;


import static Jowil.CSVHandler.NOT_AVAILABLE;
import static java.util.Arrays.asList;

public class GroupsController  extends Controller{


    //components

    @FXML
    Separator midSeparator;

    @FXML
    JFXTreeTableView groupsTable;
    
    @FXML
    VBox groupsTableVbox; 
    
    @FXML
    VBox choicesTreeVBox;


    static TreeView choicesTreeView=new TreeView();
    
    @FXML
    Button manualButton;

    @FXML
    Label groupsLabel;

    @FXML
    Label treeLabel;


    @FXML
    Label  infoLabel;

    @FXML
    VBox infoContainerVBox;



    @FXML
    VBox infoVBox;

    @FXML
    ScrollPane infoScrollPane;








    static final String Q_COUNT_COL_TITLE="Questions Count";

    JFXTreeTableColumn<Group,String> groupNamesCol = new JFXTreeTableColumn<>("Group");
    JFXTreeTableColumn<Group,String> qCountCol = new JFXTreeTableColumn<>(Q_COUNT_COL_TITLE);




    //data fields
    ObservableList<Group> tableGroups = FXCollections.observableArrayList();
    ArrayList<Group> detectedGroups;
    private static ArrayList<ArrayList<Boolean>> isPossible;
    private static ArrayList<Group> treeViewGroups;
    private static HashMap<String,Integer> groupsIndices=new HashMap<String,Integer>();

    Label [] infoKeyLabels={new Label("Number of Objective Questions"),new Label("Number of Objective Groups"),
            new Label("Number of Subjective Questions"),new Label("Number of Students"),
            new Label("Number of Forms"),new Label("Identifier Column"),new Label("Form Column")};;
    ArrayList<Label>infoValueLabels;

    //main methods
    GroupsController(Controller back){

        super("ViewGroupsAndSubjs.fxml","Groups",1.25,1.25,true,back,"2.png",1,(isQuestMode?"Questionnaire":"Test")+" Overview",resX*750/1280,0);
        CSVHandler.initQuestionsChoices();
    }


    @Override
    protected void initComponents() {

//        groupsLabel.setAlignment(Pos.CENTER);
//        treeLabel.setAlignment(Pos.CENTER);
        initGroupsTableVBox();
        initManualButton();
        initTreeView();
        initInfoVBox();
        midSeparator.setVisible(true);
        //initInfoVBox();
        choicesTreeVBox.getChildren().add(choicesTreeView);

        backButton.setOnMouseClicked(event -> {

            rootPane.requestFocus();
            if((Boolean)generalPrefsJson.get(ASK_CONTINUE_FILE_CONFIG_JSON_KEY)){
                if(!showConfirmBack())
                    return;
            }

            back.showWindow();
            stage.close();
        });

        if(isQuestMode)
            nextButton.setText("Finish");

        nextButton.setOnMouseClicked(event -> {

            rootPane.requestFocus();

            if(!saveTreeViewGroups())
                return;

            CSVHandler.updateQuestionsChoices();



            if(next==null || isContentEdited) { //if first time or edit manually has been pressed
                next = getNextController();
                next.startWindow();
            }
            else
                next.showWindow();

            isContentEdited=false;
            if(!isQuestMode)
                stage.close();

        });

        Font titleFont=new Font("Arial", headersFontSize);
        treeLabel.setFont(titleFont);
        groupsLabel.setFont(titleFont);
        infoLabel.setFont(titleFont);
        if(isQuestMode)
            infoLabel.setText("Questionnaire Info");

    }

    @Override
    protected void updateSizes(){
        super.updateSizes();
        groupsTableVbox.setSpacing(resYToPixels(0.02));
        groupsTableVbox.setLayoutY(rootHeightToPixels(0.05));
        groupsTableVbox.setPadding(new Insets(0, 0, 0, 0));
        //groupsLabel.setPrefHeight(rootHeightToPixels(0.05));
        choicesTreeVBox.setSpacing(groupsTableVbox.getSpacing());
        choicesTreeVBox.setLayoutY(groupsTableVbox.getLayoutY());
        choicesTreeVBox.setPadding(groupsTableVbox.getPadding());
        groupsTable.setPrefHeight(rootHeightToPixels(0.67));
        groupsTable.setPrefWidth(rootWidthToPixels(0.27));
        groupsLabel.setPrefWidth(groupsTable.getPrefWidth());
        treeLabel.setPrefWidth(groupsTable.getPrefWidth());
        choicesTreeView.setPrefHeight(groupsTable.getPrefHeight());
        choicesTreeView.setPrefWidth(groupsTable.getPrefWidth());




        infoScrollPane.setPrefWidth(groupsTable.getPrefWidth());
        infoScrollPane.setPrefHeight(groupsTable.getPrefHeight());
        infoVBox.setPrefWidth(groupsTable.getPrefWidth()*0.98);

        infoVBox.setMinWidth(resX*0.72*0.27);

        //gridPaneContainer.setMaxWidth(infoVBox.getPrefWidth());

        manualButton.setPrefHeight(navHeight);
        manualButton.setLayoutX(buttonsHbox.getLayoutX());
        manualButton.setLayoutY(buttonsHbox.getLayoutY()+buttonsHbox.getPadding().getTop());
//        double tablesShift=0.13;
//        groupsTableVbox.setLayoutX(rootWidthToPixels(tablesShift));
//        choicesTreeVBox.setLayoutX(rootWidthToPixels(1-tablesShift)-choicesTreeView.getPrefWidth());

        groupsTableVbox.setLayoutX(buttonsHbox.getLayoutX());
        choicesTreeVBox.setLayoutX(groupsTableVbox.getLayoutX()+groupsTable.getPrefWidth()+rootWidth*0.06);

        midSeparator.setLayoutX(rootWidthToPixels(0.665));
        midSeparator.setLayoutY(rootHeight*0.03);
        midSeparator.setPrefHeight(rootHeightToPixels(0.8));

        infoContainerVBox.setLayoutX(buttonsHbox.getLayoutX()+buttonsHbox.getPrefWidth()-infoScrollPane.getPrefWidth());
        infoContainerVBox.setLayoutY(groupsTableVbox.getLayoutY());
        infoContainerVBox.setSpacing(groupsTableVbox.getSpacing());



        //infoVBox.setPrefWidth(midSeparator.getLayoutX()-gridPaneContainer.getLayoutX()-rootWidthToPixels(0.02));


        manualButton.toFront(); //so that it's clickable and not the buttons hbox

        if(!isTranslationMode){ // to prevent questions count title from being replaced by bubbles
            if(rootWidth<resX*0.7)
                qCountCol.setText("Q. Count");
            else
                qCountCol.setText(Q_COUNT_COL_TITLE);
        }




//
//        if(rootWidth<origSceneWidth) {
//            for (int i = 0; i < infoKeyLabels.length; i++) {
//                Font labelsFont = new Font(rootWidth*12/1280*1.25);
//                infoKeyLabels[i].setFont(labelsFont);
//                infoValueLabels.get(i).setFont(labelsFont);
//            }
//        }
//

    }

    @Override
    protected Controller getNextController() {
        return isQuestMode?new QuestionnaireReportsController(this):new WeightsController(this);
    }


    @Override
    protected void buildComponentsGraph(){
        super.buildComponentsGraph();
        groupsTable.getColumns().addAll(groupNamesCol,qCountCol);

    }

    @Override
    protected void saveChanges() {


    }

    @Override
    protected void stabalizeTables(){
        //disableTableDrag(groupsTable);

    }

    @Override
    public void startWindow(){
        super.startWindow();

//        for (Node n: groupsTable.lookupAll(".column-header"))
//            n.setStyle("-fx-font-size:"+resX*14/1280);
//
//        for (Node n: groupsTable.lookupAll(".tree-table-cell"))
//            n.setStyle("-fx-font-size:"+resX*14/1280);





    }

    /////static utility methods

    public static void addToGroup(String groupName){

        int i=0;
        for(Group group: treeViewGroups){  //group names are unique
            if(group.getCleanedName().equals(groupName)) {
                if(group.isBinaryResponse())
                    return;
                group.addAnswerToEnd();
                isPossible.get(i).add(true);
                TreeItem groupItem=(TreeItem)(choicesTreeView.getRoot().getChildren().get(i));
                TreeItem rangeItem=(TreeItem)groupItem.getChildren().get(0);
                rangeItem.getChildren().add(new TreeItem<>(group.getPossibleAnswers().get(group.getPossibleAnswers().size()-1)));
                break;
            }
            i++;
        }
    }

    public static void deleteFromGroup(String groupName,String deletedChoice){

        int i=indexOfGroup(groupName);
        isPossible.get(i).set(treeViewGroups.get(i).getPossibleAnswers().indexOf(deletedChoice),false);

    }

    public boolean saveTreeViewGroups(){

        ArrayList<Group> newDetectedGroups=new ArrayList<>();

        for(Group group:treeViewGroups) //copy by value using copy constructor to have different possible answers array
            newDetectedGroups.add(new Group(group));


        for(int i=0;i<newDetectedGroups.size();i++) {
            String invalid=null;
            if ((invalid=newDetectedGroups.get(i).updatePossibleAnswers(isPossible.get(i),isQuestMode))!=null ) {
                    showAlertAndWait(Alert.AlertType.ERROR, stage.getOwner(), "Groups Choices Error",
                            constructMessage("Removing choice", " \"" + invalid + "\" ", "in group", " \"" + newDetectedGroups.get(i).getCleanedName() +
                                    "\" ", "is not allowed.", " \"" + invalid + "\" ", "is set as the correct answer for one or more of the questions in this group."));
                    return false;

            }
        }

        CSVHandler.setDetectedGroups(newDetectedGroups);
        return true;
    }

    public static boolean isChoicePossible(String groupName,String choiceName){

        int i=indexOfGroup(groupName);

        return isPossible.get(i).get(treeViewGroups.get(i).getPossibleAnswers().indexOf(choiceName));
    }


    public static void restoreToGroup(String groupName,String restoredChoice){

        int i=indexOfGroup(groupName);

        isPossible.get(i).set(treeViewGroups.get(i).getPossibleAnswers().indexOf(restoredChoice),true);


    }

    public static int getFirstPossible(String groupName){

        int i=indexOfGroup(groupName);
        return isPossible.get(i).indexOf(true);

    }

    public static int getLastPossible(String groupName){

        int i=indexOfGroup(groupName);
        return isPossible.get(i).lastIndexOf(true);
    }


    public void populateTreeViewGroups(){
        treeViewGroups=new ArrayList<Group>();

        int i=0;
        for(Group group:CSVHandler.getDetectedGroups()){ //copy by value using copy constructor.
            groupsIndices.put(group.getCleanedName(),i);
            treeViewGroups.add(new Group(group));
            i++;
        }

    }




    //helper methods
    private void initGroupsTableVBox(){


        groupNamesCol.setCellValueFactory((TreeTableColumn.CellDataFeatures<Group, String> param) -> {
           return param.getValue().getValue().getCleanedNameProp();
        });


        groupNamesCol.setCellFactory((TreeTableColumn<Group, String> param) -> new GenericEditableTreeTableCell<>(
                new TextFieldEditorBuilder()));



        qCountCol.setCellValueFactory((TreeTableColumn.CellDataFeatures<Group, String> param) -> {
            return param.getValue().getValue().getObsQcountProp();
        });

        qCountCol.setCellFactory((TreeTableColumn<Group, String> param) -> new GenericEditableTreeTableCell<>(
                new TextFieldEditorBuilder()));


        final TreeItem<Group> root = new RecursiveTreeItem<>(tableGroups, RecursiveTreeObject::getChildren);

        groupsTable.setRoot(root);
        groupsTable.setShowRoot(false);
//        groupsTable.applyCss();
//        groupsTable.lookup(".column-header").setStyle("-fx-font-size:15;");




//        groupsTable.skinProperty().addListener((a, b, newSkin) -> {
//            TableHeaderRow headerRow = ((TableViewSkinBase) newSkin).getTableHeaderRow();
//            Set<Node> nodes=headerRow.lookupAll(".label");
//            Iterator<Node> it= nodes.iterator();
//            while(it.hasNext()){
//                it.next().setStyle("-fx-font-size:20");
//            }
//
//        });

        detectedGroups=CSVHandler.getDetectedGroups();

        for(int i=0;i<detectedGroups.size();i++)
            tableGroups.add(detectedGroups.get(i));




        groupsTable.setEditable(false);


        groupsTable.setColumnResizePolicy(TreeTableView.CONSTRAINED_RESIZE_POLICY);
        //groupsTable.setSelectionModel(null);
        //groupsTable.setStyle("-fx-border-color:#1E90FF");
        groupNamesCol.setSortable(false);

        groupsTable.setOnMousePressed(t->{
//            for(Object pos:groupsTable.getSelectionModel().getSelectedIndices())
//                groupsTable.getSelectionModel().clearSelection((Integer)pos);

            });
        qCountCol.setSortable(false);
        qCountCol.setEditable(false);
        //groupsTable.setStyle("-fx-font-size:"+Double.toString(resX/106.7));

    }


    private void initInfoVBox() {

        double vSpace=resY*0.017;
        Insets leftLabelsPadding=new Insets(vSpace,0,vSpace,resX*0.004);
        Insets rightLabelsPadding=new Insets(vSpace,0,vSpace,0);




        infoValueLabels=new ArrayList<>();


        String [] values={Integer.toString(CSVHandler.getDetectedQHeaders().size()),
                Integer.toString(CSVHandler.getDetectedGroups().size()),Integer.toString(CSVHandler.getSubjQuestionsCount()),
                Integer.toString(Statistics.getStudentAnswers().size()),Integer.toString(CSVHandler.getFormsCount()),
        CSVHandler.getIdentifierColStartIndex()==NOT_AVAILABLE?"None":Statistics.getIdentifierName(),
                selectedFormColName};



        infoVBox.setStyle("-fx-background-color:transparent");
        infoScrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        infoScrollPane.focusedProperty().addListener((observable, oldValue, newValue) -> {
            if (newValue)
                rootPane.requestFocus();
        });

//        Label objCountLabel=new Label("Number of Objective Questions");
//        Label objGroupsCount=new Label("Number of Objective Groups");
//        Label subjCountLabel=new Label("Number of Subjective Questions");
//        Label studentsCountLabel=new Label("Number of Students");
//        Label formsCountLabel=new Label("Number of Forms");
//        Label identifierColLabel=new Label("Identifier Column");
//        Label formsColLabel=new Label("Form Column");

        //infoVBox.setAlignment(Pos.CENTER);


        Font labelsFont=new Font(resX*12/1280);

        for (int i=0;i<infoKeyLabels.length;i++){
            AnchorPane anchorPane=new AnchorPane();
            anchorPane.setStyle("-fx-background-color:transparent");
            infoVBox.getChildren().add(anchorPane);

            infoKeyLabels[i].setPadding(leftLabelsPadding);

            infoKeyLabels[i].setFont(labelsFont);

            infoKeyLabels[i].setStyle("-fx-font-weight: bold;");
            Label valueLabel=new Label(values[i]);
            infoValueLabels.add(valueLabel);
            valueLabel.setPadding(rightLabelsPadding);
            valueLabel.setAlignment(Pos.CENTER);
            valueLabel.setPrefWidth(resX*0.05);
            valueLabel.setFont(labelsFont);
            anchorPane.getChildren().addAll(infoKeyLabels[i],valueLabel);
            AnchorPane.setRightAnchor(valueLabel,resX*0.015);
        }








        infoScrollPane.setStyle("-fx-border-width:1;-fx-border-color:#A9A9A9;-fx-background:white;-fx-border-color:derive(-fx-background,-20%);");



    }

    private void initManualButton(){
        manualButton.getStyleClass().add("BlueJFXButton");
        manualButton.setOnMouseClicked(t->{
            new ManualModeController(this).startWindow();
        });
    }



    private void initTreeView(){


        populateTreeViewGroups();

        isPossible=new ArrayList<ArrayList<Boolean>>();

        for(Group group: treeViewGroups){  //init isPossible
            ArrayList<Boolean> possibles=new ArrayList<Boolean>();
            for(int i=0;i<group.getPossibleAnswers().size();i++)
                possibles.add(true);
            isPossible.add(possibles);
        }


        choicesTreeView.setEditable(true);

        choicesTreeView.setCellFactory(t->new TreeViewCustomCell());

        constructChoicesTreeView(choicesTreeView);



        choicesTreeView.setShowRoot(false);
    }
    
    private static void constructChoicesTreeView(TreeView treeView){


        TreeItem<String> rootItem =
                new TreeItem<String>("Groups");

        rootItem.setExpanded(true);
        treeView.setRoot(rootItem);

        int i=0;

        for(Group group: treeViewGroups){
            TreeItem groupItem=new TreeItem<>(group.getCleanedName());
            groupItem.setExpanded(true);
            rootItem.getChildren().add(groupItem);
            TreeItem rangeItem=new TreeItem();
            groupItem.getChildren().add(rangeItem);
                //treeView.scrollTo(i);

            for(String answer: group.getPossibleAnswers())
                rangeItem.getChildren().add(new TreeItem<>(answer));

            i++;
        }

    }



    private static int indexOfGroup(String groupName){

        return groupsIndices.get(groupName);
    }

    private boolean showConfirmBack() {

        Dialog dialog= new Dialog();
        dialog.setTitle("Continue to File Configuration");
        dialog.setHeaderText(null);
        dialog.setGraphic(new ImageView("Images/Error_48px.png"));
        Label label=new Label("Going back to file configuration may reset some of the" +
                "\nchanges you made to this project. Are you sure you want\nto continue to file configuration?");

        JFXCheckBox checkBox=new JFXCheckBox("Don't ask me again",14);
        checkBox.selectedProperty().addListener((observable, oldValue, newValue) -> {
            generalPrefsJson.put(ASK_CONTINUE_FILE_CONFIG_JSON_KEY,!newValue);
        });
        checkBox.setStyle("-jfx-checked-color: #095c90;");
        VBox vbox=new VBox(8,label,checkBox);
        dialog.getDialogPane().getButtonTypes().setAll(ButtonType.YES,ButtonType.NO);
        dialog.getDialogPane().lookupButton(ButtonType.YES).setStyle("-fx-border-color: #095c90;-fx-text-fill:#095c90;");
        dialog.getDialogPane().setContent(vbox);
        processDialog(dialog);
        Optional<ButtonType> option = dialog.showAndWait();
        return option.isPresent() && option.get() == ButtonType.YES;
    }
}



