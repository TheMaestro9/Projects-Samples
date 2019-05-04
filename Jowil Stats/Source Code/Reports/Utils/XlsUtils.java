package Jowil.Reports.Utils;

import Jowil.Reports.ReportsHandler;
import org.apache.poi.hssf.usermodel.*;
import org.apache.poi.hssf.util.HSSFColor;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.ss.usermodel.Color;
import org.apache.poi.ss.util.CellRangeAddress;
import org.apache.poi.ss.util.CellUtil;
import org.apache.poi.util.IOUtils;
import org.apache.poi.xwpf.usermodel.XWPFTableCell;

import java.awt.*;
import java.io.*;
import java.util.ArrayList;

public class XlsUtils {


    public static HSSFWorkbook workbook ;
    public static HSSFSheet sheet ;
    public static int pageWidth ;
    public final static int TABLE_ROW_HEIGHT = 400 ;
    final static String REPORTS_COLOR = "095c90" ;
    public static int nextColorIndex;
    public static int nextColorStartIndex = 40 ;
    public static final int  DEFAULT_NUMBER_OF_LINES_AFTER_TABLE = 2 ;
    public static HSSFFont boldFont ;
    private static HSSFCellStyle tableTitleStyle ;
    private static HSSFCellStyle defaultTableCellStyle ;
    private static HSSFCellStyle defaultCellStyle ;


    private static final int ROW_START_INDEX = 1 ;
    public static int lastRowIndex = ROW_START_INDEX ;

    public static final int REPORT_COL_START_INDEX =1;
    public static final int  PAGE_COl_PADDING = 1;
    public static final int  DEFAULT_TABLE_COl_STARTING_INDEX = REPORT_COL_START_INDEX+PAGE_COl_PADDING ;
    public static int REPORT_COL_END_INDEX ;
    //    public static final int
    public static void createXls (int width) {
        workbook = new HSSFWorkbook() ;
        sheet = workbook.createSheet();

        pageWidth = width ;
        nextColorIndex = nextColorStartIndex ;
        REPORT_COL_END_INDEX = REPORT_COL_START_INDEX + pageWidth - 1 ;
        boldFont= workbook.createFont();
        boldFont.setBold(true);

        defaultTableCellStyle= getDefaltTableCellStyle() ;
        defaultCellStyle = getDefaltCellStyle() ;
        tableTitleStyle = getTableTitleStyle() ;
//        sHSSFCellStyle defaultColumnStyle = getDefaltColumnStyle();
//        for(int i = 0 ; i < pageWidth ; i ++  ) {
//            sheet.setDefaultColumnStyle(i ,defaultColumnStyle );
//        }
    }

    private static HSSFCellStyle getTableTitleStyle() {

        HSSFCellStyle style = workbook.createCellStyle() ;
        style.setVerticalAlignment(VerticalAlignment.CENTER);
        HSSFFont font = workbook.createFont() ;
        font.setBold(true);
        font.setFontHeightInPoints((short)11);
        style.setFont(font);

        return style ;
    }

    public static short getColor(String color) {

        byte r =(byte) Integer.parseInt(color.substring(0 , 2) , 16) ;
        byte g =(byte)Integer.parseInt(color.substring(2, 4) , 16) ;
        byte b = (byte)Integer.parseInt(color.substring(4 , 6) , 16) ;

        HSSFPalette palette = workbook.getCustomPalette();


        HSSFColor myColor  = palette.findColor(r,g,b) ;
        if(myColor == null) {
            // get the color which most closely matches the color you want to use
            palette.setColorAtIndex((short) nextColorIndex, r, g, b);
            myColor= palette.getColor(nextColorIndex++) ;
        }
        // get the palette index of that color
        return  myColor.getIndex() ;
    }

    public static HSSFRow createRow(int rowIndex){
        HSSFRow row = sheet.getRow(rowIndex);
        if(row == null)
            row = sheet.createRow(rowIndex) ;
        return row ;
    }
    public static HSSFCell createCell (HSSFRow row , int cellIndex) {
       HSSFCell cell =  row.getCell(cellIndex) ;
       if(cell == null)
           cell = row.createCell(cellIndex) ;
       return cell ;
    }


    public static void addTableTitle(String title ) {
        addTableTitle(title , DEFAULT_TABLE_COl_STARTING_INDEX , true);
    }
    public static void addTableTitle(String title , int colIndex ,  boolean incrementLastRow) {
//        HSSFRow row  = sheet.createRow(lastRowIndex);
        HSSFRow row = createRow(lastRowIndex) ;
        HSSFCell cell = row.createCell(colIndex);
        cell.setCellValue(title);
        cell.setCellStyle(tableTitleStyle);
        if(incrementLastRow)
            lastRowIndex+=1 ;
    }

    public static void addTitle(String title ) {
        addTitle(title , 3);
    }

    public static void addTitle(String title  , int numberOfLinesBelow) {

        // merge cells
        sheet.addMergedRegion(
                new CellRangeAddress(
                        lastRowIndex, //first row (0-based)
                        lastRowIndex, //last row (0-based)
                        REPORT_COL_START_INDEX, //first column (0-based)
                        REPORT_COL_END_INDEX //last column (0-based)
                )
        );

        HSSFRow row = sheet.createRow(lastRowIndex);
        row.setHeight((short) 500);
        HSSFCell cell = row.createCell(REPORT_COL_START_INDEX);
        cell.setCellValue(title);

//        HSSFRow row2 = sheet.createRow(10) ;
//        HSSFCell cell2 = row2.createCell(0);


        HSSFCellStyle titleStyle = defaultCellStyle;

        //set background color
        titleStyle.setFillForegroundColor(getColor(REPORTS_COLOR));
        titleStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);
        cell.setCellStyle(titleStyle);

        //set font
        HSSFFont font= workbook.createFont();
        font.setBold(true);
        font.setFontHeightInPoints((short)18);
        font.setColor((short) 9 ); // white color index in palette
        titleStyle.setFont(font);

        lastRowIndex += numberOfLinesBelow + 1 ;
    }

    public static HSSFCellStyle getDefaltTableCellStyle(){
        HSSFCellStyle defaultTableCellStyle = workbook.createCellStyle();

        defaultTableCellStyle.setAlignment(HorizontalAlignment.CENTER);
        defaultTableCellStyle.setVerticalAlignment(VerticalAlignment.CENTER);
        defaultTableCellStyle.setBorderTop(BorderStyle.THIN);
        defaultTableCellStyle.setBorderBottom(BorderStyle.THIN);
        defaultTableCellStyle.setBorderRight(BorderStyle.THIN);
        defaultTableCellStyle.setBorderLeft(BorderStyle.THIN);
        return  defaultTableCellStyle ;
    }

    public static HSSFCellStyle getLRTableCellStyle(){
        HSSFCellStyle defaultTableCellStyle = workbook.createCellStyle();

        defaultTableCellStyle.setVerticalAlignment(VerticalAlignment.CENTER);
        return  defaultTableCellStyle ;
    }

    public static HSSFCellStyle getDefaltCellStyle(){
        HSSFCellStyle defaultColStyle = workbook.createCellStyle();
        defaultColStyle.setAlignment(HorizontalAlignment.CENTER);
        defaultColStyle.setVerticalAlignment(VerticalAlignment.CENTER);
        return defaultColStyle;
    }

    public static void parseCellClass(HSSFCell cell , String cellClassesString) {
        String [] cellClasses = cellClassesString.split(" ") ;
        ArrayList<String> cellClassesList = new ArrayList<>();
        for(int i =0 ; i < cellClasses.length ; i++)
            cellClassesList.add(cellClasses[i]) ;

        if(cellClassesList.contains("red")) {
            CellUtil.setCellStyleProperty(cell, CellUtil.FILL_FOREGROUND_COLOR, getColor("f87878"));
            CellUtil.setCellStyleProperty(cell, CellUtil.FILL_PATTERN, FillPatternType.SOLID_FOREGROUND);
        }
        if(cellClassesList.contains("gold")) {
            CellUtil.setCellStyleProperty(cell, CellUtil.FILL_FOREGROUND_COLOR, getColor("ffe44d"));
            CellUtil.setCellStyleProperty(cell, CellUtil.FILL_PATTERN, FillPatternType.SOLID_FOREGROUND);
        }
        if(cellClassesList.contains("green")) {
            CellUtil.setCellStyleProperty(cell, CellUtil.FILL_FOREGROUND_COLOR, getColor("71e08d"));
            CellUtil.setCellStyleProperty(cell, CellUtil.FILL_PATTERN, FillPatternType.SOLID_FOREGROUND);
        }

        if(cellClassesList.contains("bold")) {
            CellUtil.setFont(cell , boldFont);
        }
    }

    private static void processCellData( HSSFCell cell ,   String cellData ) throws IOException {
        String cellText = cellData ; // the normal case no classes
        if(cellData.length()>7 && cellData.substring(0 , 5).equals("<<img")){
            String [] parts = cellData.split(">>");
            String imgPath = parts[1] ;
//            String[] parts2 = parts[0].split(",") ;
//            int imgWidth = Integer.valueOf(parts2[1]) ;
//            int imgHeight = Integer.valueOf(parts2[2])  ;
            addPictureToCell(imgPath , cell);
            return;
        }else if(cellData.contains(";")) {
            String [] parts = cellData.split(";");
            cellText = parts[0] ;
            String cellClass= parts[1] ;
            parseCellClass(cell , cellClass);
        }
        cell.setCellValue(cellText);
    }

    public static void addTableAlignCenter( ArrayList<ArrayList<String>> table) throws IOException {
        addTableAlignCenter(table, DEFAULT_TABLE_COl_STARTING_INDEX , "" , DEFAULT_NUMBER_OF_LINES_AFTER_TABLE );
    }
    public static void addTableAlignCenter( ArrayList<ArrayList<String>> table , int colStartIndex  , String title , int numberOfLinesAfterTable) throws IOException {


        HSSFRow row ;
        int extraRow = 0 ;
        if(!title.equals("")) {
            extraRow  = 1 ;
            addTableTitle(title , colStartIndex , false);
        }

        HSSFCellStyle cellStyle = defaultTableCellStyle ;

        for (int rowIndex = 0 ; rowIndex < table.size() ; rowIndex++) {
            ArrayList<String> tableRow = table.get(rowIndex);
            row  = createRow(lastRowIndex + extraRow +rowIndex);
            row.setHeight((short)TABLE_ROW_HEIGHT);
            for(int colIndex = 0 ; colIndex < tableRow.size(); colIndex++) {
                HSSFCell cell = row.createCell(colStartIndex+colIndex);
                cell.setCellStyle(cellStyle);
                processCellData(cell ,tableRow.get(colIndex));
                if(rowIndex==0) {
                    CellUtil.setFont(cell , boldFont);
                }

            }

        }
            if(numberOfLinesAfterTable!=0)
                lastRowIndex += table.size() + numberOfLinesAfterTable + extraRow ;
    }

    public static void addTableAlignLR( ArrayList<ArrayList<String>> table , String title  ) {
        addTableAlignLR( table,  title , DEFAULT_NUMBER_OF_LINES_AFTER_TABLE);
    }
    public static void addTableAlignLR( ArrayList<ArrayList<String>> table , String title  , int numberOfLinesAfterTable) {
        // add title
        HSSFRow row ;
        int extraRow = 0;
        if(!title.equals("")) {
          extraRow  = 1 ;
         addTableTitle(title,DEFAULT_TABLE_COl_STARTING_INDEX , false);
        }

        HSSFCellStyle cellStyle = getLRTableCellStyle();

        for (int rowIndex = 0 ; rowIndex < table.size() ; rowIndex++) {


            ArrayList<String> tableRow = table.get(rowIndex);
            row  =sheet.createRow(lastRowIndex +rowIndex+extraRow);
            row.setHeight((short)TABLE_ROW_HEIGHT);
            for(int colIndex = 0 ; colIndex < tableRow.size(); colIndex++) {

                HSSFCell cell = row.createCell(DEFAULT_TABLE_COl_STARTING_INDEX+colIndex);
                cell.setCellValue(tableRow.get(colIndex)) ;
                cell.setCellStyle(cellStyle);

                if(rowIndex==0) {
                    CellUtil.setCellStyleProperty(cell , CellUtil.BORDER_TOP ,BorderStyle.MEDIUM );
                }

                if(colIndex%2==0)
                    CellUtil.setAlignment(cell , HorizontalAlignment.LEFT);
                else
                    CellUtil.setAlignment(cell , HorizontalAlignment.RIGHT);


            }

        }

        if(numberOfLinesAfterTable!=0)
            lastRowIndex += table.size() + numberOfLinesAfterTable+extraRow;
    }

    public static void addReport8Table(ArrayList<ArrayList<String>> table , int numberOfTableCols){

        HSSFCellStyle cellStyle = getLRTableCellStyle();

        int numberOfFirstColCells = (int)Math.ceil(numberOfTableCols/2) ;
        int numberOfSecondColCells = numberOfTableCols - numberOfFirstColCells;
        for (int rowIndex = 0 ; rowIndex < table.size() ; rowIndex++) {
            sheet.addMergedRegion(
                    new CellRangeAddress(
                            lastRowIndex +rowIndex, //first row (0-based)
                            lastRowIndex +rowIndex, //last row (0-based)
                            DEFAULT_TABLE_COl_STARTING_INDEX, //first column (0-based)
                            DEFAULT_TABLE_COl_STARTING_INDEX+numberOfFirstColCells-1 //last column (0-based)
                    )
            );
            sheet.addMergedRegion(
                    new CellRangeAddress(
                            lastRowIndex +rowIndex, //first row (0-based)
                            lastRowIndex +rowIndex, //last row (0-based)
                            DEFAULT_TABLE_COl_STARTING_INDEX+numberOfFirstColCells, //first column (0-based)
                            DEFAULT_TABLE_COl_STARTING_INDEX+numberOfFirstColCells+numberOfSecondColCells-1 //last column (0-based)
                    )
            );
            ArrayList<String> tableRow = table.get(rowIndex);
            HSSFRow row  =sheet.createRow(lastRowIndex +rowIndex);
            row.setHeight((short)TABLE_ROW_HEIGHT);
            for(int colIndex = 0 ; colIndex < tableRow.size(); colIndex++) {

                HSSFCell cell = row.createCell(DEFAULT_TABLE_COl_STARTING_INDEX+colIndex+((numberOfFirstColCells-1)*colIndex));
                cell.setCellValue(tableRow.get(colIndex)) ;
                cell.setCellStyle(cellStyle);

                if(colIndex%2==0)
                    CellUtil.setAlignment(cell , HorizontalAlignment.LEFT);
                else
                    CellUtil.setAlignment(cell , HorizontalAlignment.RIGHT);
            }
            if(rowIndex==0) {
                for(int i = 0 ; i < numberOfTableCols ; i++) {
                    HSSFCell cell = row.getCell(DEFAULT_TABLE_COl_STARTING_INDEX+ i) ;
//                    CellStyle style = cell.getCellStyle() ;
                    if(cell == null) {
                        cell = row.createCell(DEFAULT_TABLE_COl_STARTING_INDEX + i);
                        cell.setCellStyle(defaultCellStyle);
                    }
                    CellUtil.setCellStyleProperty(cell, CellUtil.BORDER_TOP, BorderStyle.MEDIUM);
                }
            }


        }

    }

    public static void addHeaderLine (String header){

        HSSFCellStyle underLineCellStyle = workbook.createCellStyle() ;
        underLineCellStyle.setBorderBottom(BorderStyle.MEDIUM);


        HSSFRow row = sheet.createRow(lastRowIndex++) ;

        for(int i = DEFAULT_TABLE_COl_STARTING_INDEX ; i <= REPORT_COL_END_INDEX - PAGE_COl_PADDING ; i++)
            row.createCell(i).setCellStyle(underLineCellStyle);

        int cellIndex= REPORT_COL_START_INDEX+ (int)Math.floor(pageWidth/2) ;
        HSSFCell cell = row.createCell(cellIndex);
        cell.setCellStyle(tableTitleStyle);
        CellUtil.setAlignment(cell , HorizontalAlignment.CENTER);
        cell.setCellValue(header);
//        row = sheet.createRow(lastRowIndex++) ;
//        for(int i = 1 ; i <pageWidth-1 ; i++);

        lastRowIndex+= 2 ;
    }


    public static void addPictureToCell(String imgPath , HSSFCell cell ) throws IOException {
        addPictureToCell(imgPath , cell.getRowIndex() , cell.getColumnIndex() , 1, 1 , 0 );
    }

    public static void addPictureToCell(String imgPath , int rowIndex  , int colIndex , int width , int height , int numberOfLinesAfterImg ) throws IOException {
        InputStream inputStream = new FileInputStream(imgPath);

        byte[] imageBytes = IOUtils.toByteArray(inputStream);

        int pictureureIdx = workbook.addPicture(imageBytes, Workbook.PICTURE_TYPE_PNG);

        inputStream.close();

        CreationHelper helper = workbook.getCreationHelper();

        HSSFPatriarch  drawing = sheet.createDrawingPatriarch();

        ClientAnchor anchor = helper.createClientAnchor();

        anchor.setCol1(colIndex);
        anchor.setRow1(rowIndex);

        HSSFPicture pic = drawing.createPicture(anchor, pictureureIdx);
            pic.resize(width , height );
        if(height>1)
            lastRowIndex += height + numberOfLinesAfterImg  ;
    }

    public static void addReportBorder(){
        BorderStyle reportBorderStyle = BorderStyle.MEDIUM;
//        HSSFCellStyle leftStyle = workbook.createCellStyle() ;
//        leftStyle.setBorderLeft(reportBorderStyle);
//
//        HSSFCellStyle rightStyle = workbook.createCellStyle() ;
//        leftStyle.setBorderRight(reportBorderStyle);
//
//        HSSFCellStyle topStyle = workbook.createCellStyle() ;
//        leftStyle.setBorderTop(reportBorderStyle);
//
//        HSSFCellStyle bottomStyle = workbook.createCellStyle() ;
//        leftStyle.setBorderBottom(reportBorderStyle);
        //loop over the rows of the document
        for(int i  = ROW_START_INDEX ; i < lastRowIndex ; i++ ) {
            HSSFRow row = createRow(i) ;
            HSSFCell cellLeft = createCell(row,REPORT_COL_START_INDEX) ;
            HSSFCell cellRight = createCell(row,REPORT_COL_END_INDEX) ;
            CellUtil.setCellStyleProperty(cellLeft,CellUtil.BORDER_LEFT , reportBorderStyle);
            CellUtil.setCellStyleProperty(cellRight, CellUtil.BORDER_RIGHT,reportBorderStyle);
        }

        HSSFRow topRow = createRow( ROW_START_INDEX);
        HSSFRow bottomRow = createRow(lastRowIndex-1) ;

        for(int i = REPORT_COL_START_INDEX ; i <= REPORT_COL_END_INDEX ; i++) {
            HSSFCell topCell = createCell(topRow ,i);
            HSSFCell bottomCell = createCell( bottomRow , i) ;
            CellUtil.setCellStyleProperty(topCell ,CellUtil.BORDER_TOP , reportBorderStyle);
            CellUtil.setCellStyleProperty(bottomCell ,CellUtil.BORDER_BOTTOM , reportBorderStyle);
        }
    }
    public static void postProcessSheet() {
        for(int i =ROW_START_INDEX ; i < lastRowIndex ; i ++) {
            HSSFRow row = createRow(i);
            row.setHeight((short)TABLE_ROW_HEIGHT);
        }
        for( int i = REPORT_COL_START_INDEX ; i <= REPORT_COL_END_INDEX ; i++ )
            sheet.autoSizeColumn(i);

        addReportBorder();
        lastRowIndex = ROW_START_INDEX ;
    }
    public static void writeXlsFile(String filePath) throws IOException {
        writeXlsFile(filePath , true);
    }
    public static void writeXlsFile(String filePath , boolean postProcess ) throws IOException {
        if(postProcess)
            postProcessSheet();
        FileOutputStream out = new FileOutputStream(new File(filePath));
        workbook.write(out);
        out.close();
    }

}
