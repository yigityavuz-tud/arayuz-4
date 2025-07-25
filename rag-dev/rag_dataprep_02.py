import os
# import json
import glob
from pathlib import Path
from collections import Counter
from unstructured.partition.pdf import partition_pdf
# from unstructured.staging.base import elements_to_json
import re


def concatenate_text_column_aware(text_elements, exclude_types=None, column_threshold=50):
    """
    Advanced version that detects columns and follows proper N-shaped reading order.
    Uses proper spacing: paragraph breaks (\n\n) between NarrativeText elements, 
    line breaks (\n) between other elements. Normalizes multiple spaces to single space.
    
    Args:
        text_elements (list): List of text elements with coordinates and type
        exclude_types (list): List of element types to exclude
        column_threshold (int): Pixel threshold to determine column boundaries
    
    Returns:
        str: Concatenated text following column-aware reading order with proper spacing
    """
    if exclude_types is None:
        exclude_types = ['footer']
    
    # Filter out excluded types
    filtered_elements = [
        elem for elem in text_elements 
        if elem.get('type', '').lower() not in [t.lower() for t in exclude_types]
    ]
    
    if not filtered_elements:
        return ""
    
    # Group elements by approximate Y position (rows)
    rows = {}
    for element in filtered_elements:
        coords = element.get('coordinates', [[0, 0]])
        if coords:
            x, y = coords[0]  # Upper-left corner
            
            # Group by approximate Y position (allowing some tolerance)
            row_key = round(y / 20) * 20  # Group by 20-pixel rows
            
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(element)
    
    # Sort rows by Y position (top to bottom)
    sorted_rows = sorted(rows.items())
    
    # Flatten elements in reading order
    ordered_elements = []
    for row_y, row_elements in sorted_rows:
        # Within each row, sort by X position (left to right)
        row_elements.sort(key=lambda elem: elem.get('coordinates', [[0, 0]])[0][0])
        ordered_elements.extend(row_elements)
    
    # Build concatenated text with proper spacing
    concatenated_text = ""
    previous_type = None
    
    for i, element in enumerate(ordered_elements):
        text = element.get('text', '').strip()
        current_type = element.get('type', '').strip()
        
        if text:
            # Normalize multiple spaces to single space within the text
            text = re.sub(r' +', ' ', text)
            
            if i > 0:  # Not the first element
                # Determine spacing based on element types
                if (previous_type == 'NarrativeText' and current_type == 'NarrativeText'):
                    # Paragraph break between two NarrativeText elements
                    concatenated_text += "\n\n"
                else:
                    # Line break for all other combinations
                    concatenated_text += "\n"
            
            concatenated_text += text
            previous_type = current_type
    
    return concatenated_text.strip()


def process_pdf_file(pdf_path, output_txt_dir):
    """
    Process a single PDF file and create corresponding text file.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_txt_dir (str): Directory to save the text file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get base filename without extension
        base_file_name = Path(pdf_path).stem
        
        print(f"Processing: {base_file_name}.pdf")
        
        # Extract elements from PDF
        elements = partition_pdf(
            filename=pdf_path,
            languages=["tur"],
            strategy="fast",
            infer_table_structure=True,
        )
        
        # Convert elements to JSON format for processing
        elements_json = []
        for element in elements:
            element_dict = element.to_dict()
            elements_json.append(element_dict)
        
        # Optional: Save JSON file for debugging (uncomment if needed)
        # json_output_path = os.path.join(output_txt_dir, f"{base_file_name}-output.json")
        # elements_to_json(elements=elements, filename=json_output_path)
        
        # Count element types (for debugging)
        type_counts = Counter(item["type"] for item in elements_json if "type" in item)
        print(f"  Element types found: {dict(type_counts)}")
        
        # Apply column-aware concatenation
        concatenated_text = concatenate_text_column_aware(elements_json)
        
        # Create output text file path
        txt_output_path = os.path.join(output_txt_dir, f"{base_file_name}.txt")
        
        # Save concatenated text to file
        with open(txt_output_path, "w", encoding="utf-8") as file:
            file.write(concatenated_text)
        
        print(f"  ✓ Successfully saved: {base_file_name}.txt")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {pdf_path}: {str(e)}")
        return False


def main():
    """
    Main function to process all PDF files in the input directory.
    """
    # Define directories
    input_pdf_dir = "C:/Users/yigit/Desktop/Enterprises/arayuz-3/okumalar-pdf/"
    output_txt_dir = "C:/Users/yigit/Desktop/Enterprises/arayuz-3/okumalar-txt/"
    
    # Create output directory if it doesn't exist
    Path(output_txt_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_pattern = os.path.join(input_pdf_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in: {input_pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process.")
    print(f"Input directory: {input_pdf_dir}")
    print(f"Output directory: {output_txt_dir}")
    print("-" * 50)
    
    # Process each PDF file
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        if process_pdf_file(pdf_file, output_txt_dir):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("-" * 50)
    print(f"Processing complete!")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(pdf_files)}")


if __name__ == "__main__":
    main()