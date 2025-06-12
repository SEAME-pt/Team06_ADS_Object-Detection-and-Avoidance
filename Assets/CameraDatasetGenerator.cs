using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Linq;

public class CameraDatasetGenerator : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera mainCamera;
    
    [Header("Signal Tags")]
    public List<string> signalTags = new List<string> { "STOP", "PASSAGEM", "VEL_50", "VEL_80", "SEMAFORO_VERMELHO", "SEMAFORO_VERDE", "SEMAFORO_LARANJA","PASSADEIRA","DANGER","CURVA" };
    
    [Header("Output Settings")]
    public string outputFolder = "Dataset";
    public int textureWidth = 1024;
    public int textureHeight = 720;
    
    [Header("Movement Settings")]
    public float moveSpeed = 5f;
    public float rotationSpeed = 100f;
    
    [Header("Preview Settings")]
    public bool showPreview = true;
    public bool showBoundingBoxes = true;
    public Color boundingBoxColor = Color.red;
    public float boundingBoxLineWidth = 2f;
    
    [Header("Validation Settings")]
    public float minBoundingBoxSize = 0.01f; // Mínimo 1% da tela
    public float maxBoundingBoxSize = 0.8f;  // Máximo 80% da tela

    private RenderTexture renderTexture;
    private int fileCounter = 0;
    private List<BoundingBoxInfo> currentBoundingBoxes = new List<BoundingBoxInfo>();
    private bool showGUI = true;
    bool existeSinalVisivel;

    [System.Serializable]
    public class BoundingBoxInfo
    {
        public string tag;
        public int classId;
        public Rect screenRect;
        public Rect normalizedRect;
        public GameObject gameObject;
        public bool isValid;
        
        public BoundingBoxInfo(string tag, int classId, Rect screenRect, Rect normalizedRect, GameObject obj)
        {
            this.tag = tag;
            this.classId = classId;
            this.screenRect = screenRect;
            this.normalizedRect = normalizedRect;
            this.gameObject = obj;
            this.isValid = true;
        }
    }

    void Start()
    {
        SetupDirectories();
        SetupCamera();
        fileCounter = GetLastFileIndex(Path.Combine(outputFolder, "images"));
    }

    void SetupDirectories()
    {
        if (!Directory.Exists(outputFolder))
            Directory.CreateDirectory(outputFolder);

        string imagesFolder = Path.Combine(outputFolder, "images");
        string labelsFolder = Path.Combine(outputFolder, "labels");

        if (!Directory.Exists(imagesFolder))
            Directory.CreateDirectory(imagesFolder);
        if (!Directory.Exists(labelsFolder))
            Directory.CreateDirectory(labelsFolder);
    }

    void SetupCamera()
    {
        renderTexture = new RenderTexture(textureWidth, textureHeight, 24, RenderTextureFormat.ARGB32);
        renderTexture.Create();
        mainCamera.targetTexture = renderTexture;
    }

    void Update()
    {
        HandleMovement();
        HandleInput();
        
        if (showBoundingBoxes)
        {
            UpdateBoundingBoxes();
        }
         existeSinalVisivel = currentBoundingBoxes.Any(box => box.isValid);
    }

    void HandleMovement()
    {
        // Movimento WASD
        float moveHorizontal = Input.GetAxis("Horizontal");
        float moveVertical = Input.GetAxis("Vertical");
        Vector3 movement = new Vector3(moveHorizontal, 0.0f, moveVertical) * moveSpeed * Time.deltaTime;
        transform.Translate(movement, Space.Self);

        // Movimento Q/E para subir/descer
        if (Input.GetKey(KeyCode.Q))
            transform.Translate(Vector3.up * moveSpeed * Time.deltaTime, Space.World);
        if (Input.GetKey(KeyCode.E))
            transform.Translate(Vector3.down * moveSpeed * Time.deltaTime, Space.World);

        // Rotação com mouse
        if (Input.GetMouseButton(1)) // Botão direito do mouse
        {
            float rotationY = Input.GetAxis("Mouse X") * rotationSpeed * Time.deltaTime;
            float rotationX = Input.GetAxis("Mouse Y") * rotationSpeed * Time.deltaTime;
            transform.Rotate(Vector3.up, rotationY, Space.World);
            transform.Rotate(Vector3.right, -rotationX, Space.Self);
        }
    }

    void HandleInput()
    {
        // Captura de imagem
        if (Input.GetKeyDown(KeyCode.Space) && existeSinalVisivel)
        {
            StartCoroutine(CaptureImageAndAnnotations());
        }
        
        // Toggle GUI
        if (Input.GetKeyDown(KeyCode.G))
        {
            showGUI = !showGUI;
        }
        
        // Toggle bounding boxes
        if (Input.GetKeyDown(KeyCode.B))
        {
            showBoundingBoxes = !showBoundingBoxes;
        }
        if (Input.GetMouseButtonUp(0) && existeSinalVisivel)
        {
         StartCoroutine(CaptureImageAndAnnotations());
        }
    }

    void UpdateBoundingBoxes()
    {
        currentBoundingBoxes.Clear();
        
        foreach (string tag in signalTags)
        {
            foreach (GameObject signal in GameObject.FindGameObjectsWithTag(tag))
            {
                BoundingBoxInfo boxInfo = CalculateBoundingBox(signal, tag);
                if (boxInfo != null)
                {
                    currentBoundingBoxes.Add(boxInfo);
                }
            }
        }
    }

    BoundingBoxInfo CalculateBoundingBox(GameObject signal, string tag)
    {
        Renderer renderer = signal.GetComponent<Renderer>();
        if (renderer == null) return null;

        Bounds bounds = renderer.bounds;
        
        // Verifica se está no frustum da câmara
        if (!GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(mainCamera), bounds))
            return null;

        // Calcula todos os vértices da bounding box 3D
        Vector3[] vertices = GetBoundingBoxVertices(bounds);
        
        // Converte para coordenadas de tela
        List<Vector2> screenPoints = new List<Vector2>();
        bool allBehindCamera = true;
        
        foreach (Vector3 vertex in vertices)
        {
            Vector3 screenPoint = mainCamera.WorldToScreenPoint(vertex);
            if (screenPoint.z > 0) // À frente da câmara
            {
                screenPoints.Add(new Vector2(screenPoint.x, screenPoint.y));
                allBehindCamera = false;
            }
        }

        if (allBehindCamera || screenPoints.Count == 0)
            return null;

        // Calcula bounding box 2D mínima
        float minX = screenPoints.Min(p => p.x);
        float minY = screenPoints.Min(p => p.y);
        float maxX = screenPoints.Max(p => p.x);
        float maxY = screenPoints.Max(p => p.y);

        // Clamp às dimensões da tela
        minX = Mathf.Clamp(minX, 0, renderTexture.width);
        minY = Mathf.Clamp(minY, 0, renderTexture.height);
        maxX = Mathf.Clamp(maxX, 0, renderTexture.width);
        maxY = Mathf.Clamp(maxY, 0, renderTexture.height);

        // Rect em coordenadas de tela (Y invertido para GUI)
        Rect screenRect = new Rect(minX, renderTexture.height - maxY, maxX - minX, maxY - minY);
        
        // Normaliza para formato YOLO [0,1]
        float width = (maxX - minX) / renderTexture.width;
        float height = (maxY - minY) / renderTexture.height;
        float x_center = (minX + (maxX - minX) / 2) / renderTexture.width;
        float y_center = 1.0f - (minY + (maxY - minY) / 2) / renderTexture.height; // YOLO usa origem no topo-esquerda
        
        Rect normalizedRect = new Rect(x_center, y_center, width, height);
        
        // Validação
        bool isValid = width >= minBoundingBoxSize && height >= minBoundingBoxSize &&
                      width <= maxBoundingBoxSize && height <= maxBoundingBoxSize;

        int classId = signalTags.IndexOf(tag);
        BoundingBoxInfo boxInfo = new BoundingBoxInfo(tag, classId, screenRect, normalizedRect, signal);
        boxInfo.isValid = isValid;
        
        return boxInfo;
    }

    Vector3[] GetBoundingBoxVertices(Bounds bounds)
    {
        Vector3[] vertices = new Vector3[8];
        vertices[0] = new Vector3(bounds.min.x, bounds.min.y, bounds.min.z);
        vertices[1] = new Vector3(bounds.min.x, bounds.min.y, bounds.max.z);
        vertices[2] = new Vector3(bounds.min.x, bounds.max.y, bounds.min.z);
        vertices[3] = new Vector3(bounds.min.x, bounds.max.y, bounds.max.z);
        vertices[4] = new Vector3(bounds.max.x, bounds.min.y, bounds.min.z);
        vertices[5] = new Vector3(bounds.max.x, bounds.min.y, bounds.max.z);
        vertices[6] = new Vector3(bounds.max.x, bounds.max.y, bounds.min.z);
        vertices[7] = new Vector3(bounds.max.x, bounds.max.y, bounds.max.z);
        return vertices;
    }

    IEnumerator CaptureImageAndAnnotations()
    {
        // Atualiza bounding boxes antes da captura
        UpdateBoundingBoxes();
        
        // Filtra apenas boxes válidas
        List<BoundingBoxInfo> validBoxes = currentBoundingBoxes.Where(box => box.isValid).ToList();
        
        if (validBoxes.Count == 0)
        {
            Debug.LogWarning("Nenhuma bounding box válida encontrada. Captura cancelada.");
            yield return null;
        }

        // Renderiza a cena
        mainCamera.targetTexture = renderTexture;
        mainCamera.Render();

        // Captura a imagem
        Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.ARGB32, false);
        RenderTexture.active = renderTexture;
        tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        tex.Apply();
        RenderTexture.active = null;

        // Salva imagem
        byte[] bytes = tex.EncodeToPNG();
        string imagesFolder = Path.Combine(outputFolder, "images");
        string labelsFolder = Path.Combine(outputFolder, "labels");
        string imagePath = Path.Combine(imagesFolder, $"{fileCounter}.png");
        File.WriteAllBytes(imagePath, bytes);
        Destroy(tex);

        // Gera anotações YOLO
        List<string> annotations = new List<string>();
        foreach (BoundingBoxInfo box in validBoxes)
        {
            string annotation = $"{box.classId} {box.normalizedRect.x:F6} {box.normalizedRect.y:F6} {box.normalizedRect.width:F6} {box.normalizedRect.height:F6}";
            annotations.Add(annotation);
        }

        // Salva anotações
        string annotationPath = Path.Combine(labelsFolder, $"{fileCounter}.txt");
        File.WriteAllLines(annotationPath, annotations);

        Debug.Log($"Capturada imagem {fileCounter}.png com {validBoxes.Count} sinais anotados");
        fileCounter++;
        yield return null;
    }

    void OnGUI()
    {
        if (!showGUI) return;

        // Preview da câmara
        if (renderTexture != null && showPreview)
        {
            float previewScale = 0.25f;
            float previewWidth = renderTexture.width * previewScale;
            float previewHeight = renderTexture.height * previewScale;
            Rect previewRect = new Rect(Screen.width - previewWidth - 10, 10, previewWidth, previewHeight);
            
            GUI.DrawTexture(previewRect, renderTexture, ScaleMode.ScaleToFit);
            
            // Desenha bounding boxes no preview
            if (showBoundingBoxes && currentBoundingBoxes.Count > 0)
            {
                DrawBoundingBoxesOnPreview(previewRect, previewScale);
            }
        }

        // Informações de debug
        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.Label($"Ficheiro: {fileCounter}");
        GUILayout.Label($"Posição: {transform.position}");
        GUILayout.Label($"Sinais visíveis: {currentBoundingBoxes.Count(box => box.isValid)}");
        
        GUILayout.Space(10);
        GUILayout.Label("Controlos:");
        GUILayout.Label("WASD - Mover");
        GUILayout.Label("Q/E - Subir/Descer");
        GUILayout.Label("Botão Direito + Mouse - Rodar");
        GUILayout.Label("Espaço - Capturar");
        GUILayout.Label("G - Toggle GUI");
        GUILayout.Label("B - Toggle Bounding Boxes");
        
        GUILayout.EndArea();

        // Lista de sinais detectados
        if (currentBoundingBoxes.Count > 0)
        {
            GUILayout.BeginArea(new Rect(Screen.width - 250, Screen.height - 150, 240, 140));
            GUILayout.Label("Sinais Detectados:");
            foreach (var box in currentBoundingBoxes)
            {
                Color oldColor = GUI.color;
                GUI.color = box.isValid ? Color.green : Color.red;
                GUILayout.Label($"{box.tag}: {box.normalizedRect.width:F3}x{box.normalizedRect.height:F3}");
                GUI.color = oldColor;
            }
            GUILayout.EndArea();
        }
    }

    void DrawBoundingBoxesOnPreview(Rect previewRect, float scale)
    {
        foreach (var box in currentBoundingBoxes)
        {
            Color oldColor = GUI.color;
            GUI.color = box.isValid ? boundingBoxColor : Color.yellow;
            
            // Converte coordenadas da bounding box para o preview
            Rect scaledBox = new Rect(
                previewRect.x + box.screenRect.x * scale,
                previewRect.y + box.screenRect.y * scale,
                box.screenRect.width * scale,
                box.screenRect.height * scale
            );
            
            DrawRect(scaledBox, GUI.color);
            GUI.color = oldColor;
        }
    }

    void DrawRect(Rect rect, Color color)
    {
        // Desenha um retângulo vazio (apenas bordas)
        Color oldColor = GUI.color;
        GUI.color = color;
        
        // Bordas
        GUI.DrawTexture(new Rect(rect.x, rect.y, rect.width, 2), Texture2D.whiteTexture); // Top
        GUI.DrawTexture(new Rect(rect.x, rect.y + rect.height - 2, rect.width, 2), Texture2D.whiteTexture); // Bottom
        GUI.DrawTexture(new Rect(rect.x, rect.y, 2, rect.height), Texture2D.whiteTexture); // Left
        GUI.DrawTexture(new Rect(rect.x + rect.width - 2, rect.y, 2, rect.height), Texture2D.whiteTexture); // Right
        
        GUI.color = oldColor;
    }

    int GetLastFileIndex(string imagesFolder)
    {
        if (!Directory.Exists(imagesFolder))
            return 0;

        int maxIndex = -1;
        string pattern = @"(\d+)\.png";

        foreach (string filePath in Directory.GetFiles(imagesFolder, "*.png"))
        {
            string fileName = Path.GetFileName(filePath);
            Match match = Regex.Match(fileName, pattern);
            if (match.Success)
            {
                int index = int.Parse(match.Groups[1].Value);
                if (index > maxIndex)
                    maxIndex = index;
            }
        }

        return maxIndex + 1;
    }

    void OnDestroy()
    {
        if (renderTexture != null)
            renderTexture.Release();
    }
}
